from upload_video_to_youtube import upload_video
from pathlib import Path


def main():
    root = Path("playbook_2024")
    to_upload = [
        # "01_offene_seite_give_n_gos",
        # "02_breakside_continuations",
        # "03_clearning",
        # "06_dishy",
        # "07_dishy_turn_inward",
        "08_away_from_sideline_in_flow"
    ]
    for x in to_upload:
        play_dir = root / x
        video_path_candidates = list(play_dir.glob("*.mp4"))
        assert (
            len(video_path_candidates) == 1
        ), f"Expected exactly one video in {play_dir}, got {video_path_candidates}"
        title = play_dir.name
        thumbnail_candidates = list(play_dir.glob("*.png"))
        thumbnail = thumbnail_candidates[0] if thumbnail_candidates else None
        upload_video(video_path_candidates[0], title, thumbnail)


if __name__ == "__main__":
    main()
