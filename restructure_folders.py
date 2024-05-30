from pathlib import Path
import subprocess
from shutil import copyfile


def main():
    root = "playbook_2024"
    output_dir = "animations"
    limit = None
    restructure_folders(root)
    # render_all_plays(root, limit=limit)
    copy_animations(root, output_dir)


def copy_animations(root: Path | str, output_dir=None):
    all_play_dirs = find_all_play_dirs(root)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    for play_dir in all_play_dirs:
        files_to_copy = list(play_dir.glob("*.mp4")) + list(play_dir.glob("*.png"))
        for file_to_copy in files_to_copy:
            new_file_path = output_dir / file_to_copy.name
            copyfile(file_to_copy, new_file_path)


def render_all_plays(root: Path | str, limit=None):
    all_play_dirs = find_all_play_dirs(root)
    for play_dir in all_play_dirs[:limit]:
        python_files = list(play_dir.glob("*.py"))
        if len(python_files) == 1:
            python_file = python_files[0]
            print(f"Rendering {python_file}")
            subprocess.run(["python", python_file.absolute()], cwd=play_dir)
        else:
            print(f"Expected one .py file in {play_dir}, got {python_files}")


_empty = object()


def find_all_play_dirs(root: Path | str):
    all_folders = list(Path(root).glob("**/"))
    all_play_dirs = sorted([x for x in all_folders if next(x.glob("*.py"), _empty) is not _empty])
    return all_play_dirs


def restructure_folders(root: Path | str):
    all_play_dirs = find_all_play_dirs(root)
    print(f"Found {len(all_play_dirs)} play dirs")
    for play_dir in all_play_dirs:
        states_dir = play_dir / "states"
        if not states_dir.exists():
            print(f"Moving states from {play_dir} to {states_dir}")
            states_dir.mkdir()
            yaml_files = list(play_dir.glob("*.yaml"))
            for yaml_file in yaml_files:
                yaml_file.rename(states_dir / yaml_file.name)
        for extension in "py mp4 png".split():
            files = list(play_dir.glob(f"*.{extension}"))
            if len(files) == 1:
                file = files[0]
                new_file_path = play_dir / f"{play_dir.name}.{extension}"
                if file != new_file_path:
                    print(f"Renaming {file} to {new_file_path}")
                    file.rename(new_file_path)
            elif len(files) > 1:
                print(f"Expected zero or one .{extension} file in {play_dir}, got {files}")


if __name__ == "__main__":
    main()
