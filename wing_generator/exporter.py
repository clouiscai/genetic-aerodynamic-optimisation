from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass
from pathlib import Path

from .config import GeneratorConfig
from .stl_mesh import write_stitched_stl_log, write_stitched_wing_files


@dataclass(frozen=True)
class CommandResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str

    @property
    def combined_output(self) -> str:
        chunks = []
        if self.stdout:
            chunks.append(self.stdout)
        if self.stderr:
            chunks.append(self.stderr)
        return "\n".join(chunks)


@dataclass(frozen=True)
class ExportOptions:
    validate_cad: bool = True
    export_stl: bool = True
    export_step: bool = True
    render_preview: bool = True


def validate_forgecad_model(design_dir: Path, config: GeneratorConfig) -> CommandResult:
    script = design_dir / "geometry.forge.js"
    forgecad = _forgecad_command()
    result = _run(
        [
            forgecad,
            "run",
            str(script),
            "--backend",
            config.forgecad.backend,
        ],
        timeout_s=config.forgecad.command_timeout_s,
    )
    _write_log(design_dir / "forgecad_run.log", result)
    return result


def export_design(design_dir: Path, config: GeneratorConfig, options: ExportOptions) -> None:
    script = design_dir / "geometry.forge.js"
    forgecad = _forgecad_command() if options.export_step or options.render_preview else None

    if options.export_stl:
        stl = design_dir / "geometry.stl"
        obj = design_dir / "geometry.obj"
        stl_todo = design_dir / "geometry.stl.todo.txt"
        # Remove any prior output FIRST so a failed export can never leave stale
        # geometry on disk (which would not match the current geometry.forge.js).
        stl.unlink(missing_ok=True)
        obj.unlink(missing_ok=True)
        stl_todo.unlink(missing_ok=True)
        try:
            design = json.loads((design_dir / "parameters.json").read_text(encoding="utf-8"))
            stats = write_stitched_wing_files(design, config, stl, obj)
            write_stitched_stl_log(design_dir / "stl_export.log", stats)
            write_stitched_stl_log(design_dir / "obj_export.log", stats)
            if (
                stats.boundary_edges
                or stats.nonmanifold_edges
                or stats.oriented_edge_conflicts
                or stats.connected_components != 1
            ):
                _write_todo(
                    stl_todo,
                    "TODO: stitched STL mesh did not pass topology checks. See stl_export.log.",
                )
        except Exception as exc:
            result = CommandResult(ok=False, returncode=1, stdout="", stderr=str(exc))
            _write_log(design_dir / "stl_export.log", result)
            _write_todo(
                stl_todo,
                "TODO: STL export failed (no geometry.stl written). See stl_export.log.",
            )
    else:
        (design_dir / "geometry.stl").unlink(missing_ok=True)
        (design_dir / "geometry.obj").unlink(missing_ok=True)
        _write_todo(
            design_dir / "geometry.stl.todo.txt",
            "STL export skipped for this run. Re-run without --skip-stl to regenerate it.",
        )

    if options.export_step:
        step = design_dir / "geometry.step"
        step_todo = design_dir / "geometry.step.todo.txt"
        step.unlink(missing_ok=True)
        step_todo.unlink(missing_ok=True)
        result = _run(
            [
                forgecad,
                "export",
                "step",
                str(script),
                "--output",
                str(step),
            ],
            timeout_s=config.forgecad.command_timeout_s,
        )
        _write_log(design_dir / "step_export.log", result)
        if not result.ok or not step.exists():
            _write_todo(
                step_todo,
                "TODO: STEP export failed or requires a ForgeCAD Pro-capable exporter. "
                "See step_export.log.",
            )
    else:
        (design_dir / "geometry.step").unlink(missing_ok=True)
        _write_todo(
            design_dir / "geometry.step.todo.txt",
            "STEP export skipped for this run. Re-run without --skip-step to regenerate it.",
        )

    if options.render_preview:
        preview = design_dir / "preview.png"
        (design_dir / "preview.png.todo.txt").unlink(missing_ok=True)
        render_args = [
            forgecad,
            "render",
            "3d",
            str(script),
            "--output",
            str(preview),
            "--camera",
            config.forgecad.preview_camera,
            "--size",
            str(config.forgecad.preview_size_px),
            "--edges",
            "thin",
            "--port",
            "5188",
        ]
        chrome_path = _find_chrome_path()
        if chrome_path:
            render_args.extend(["--chrome-path", chrome_path])
        
        result = _run(render_args, timeout_s=config.forgecad.command_timeout_s)
        _write_log(design_dir / "preview_render.log", result)
        if not result.ok:
            _write_todo(
                design_dir / "preview.png.todo.txt",
                "TODO: preview render failed. See preview_render.log.",
            )
    else:
        (design_dir / "preview.png").unlink(missing_ok=True)
        _write_todo(
            design_dir / "preview.png.todo.txt",
            "Preview render skipped for this run. Re-run without --skip-preview to regenerate it.",
        )


def _run(args: list[str], timeout_s: int) -> CommandResult:
    completed = subprocess.run(
        args,
        text=True,
        capture_output=True,
        timeout=timeout_s,
        check=False,
    )
    return CommandResult(
        ok=completed.returncode == 0,
        returncode=completed.returncode,
        stdout=completed.stdout,
        stderr=completed.stderr,
    )


def _forgecad_command() -> str:
    command = shutil.which("forgecad") or shutil.which("forgecad.cmd")
    if command is None:
        raise FileNotFoundError(
            "ForgeCAD CLI was not found on PATH. Install it with `npm install -g forgecad`."
        )
    return command


def _find_chrome_path() -> str | None:
    import os
    env_path = os.environ.get("CHROME_PATH")
    if env_path and Path(env_path).is_file():
        return env_path
    
    for name in ("chrome", "google-chrome", "google-chrome-stable", "chromium"):
        cmd = shutil.which(name)
        if cmd:
            return cmd

    win_paths = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    ]
    for p in win_paths:
        if Path(p).is_file():
            return p
            
    return None


def _write_log(path: Path, result: CommandResult) -> None:
    path.write_text(
        f"returncode: {result.returncode}\n\nSTDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}\n",
        encoding="utf-8",
    )


def _write_todo(path: Path, message: str) -> None:
    path.write_text(message + "\n", encoding="utf-8")
