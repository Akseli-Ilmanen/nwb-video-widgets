"""Tests for get_videos() — mapping session time to video files and local times."""

import shutil
from pathlib import Path

import pytest
from pynwb import NWBHDF5IO, read_nwb
from pynwb.image import ImageSeries
from pynwb.testing.mock.file import mock_NWBFile

from nwb_video_widgets._utils import _get_video_duration, get_videos
from tests.conftest import STUB_H264_PATH

# --- Helpers ---


def _save_and_load(nwbfile, nwb_path):
    """Write an NWB file to disk and re-read it so read_io.source is set."""
    with NWBHDF5IO(str(nwb_path), "w") as io:
        io.write(nwbfile)
    return read_nwb(str(nwb_path))


def _make_multi_segment_nwbfile(tmp_path, rate=30.0, starting_time=0.0):
    """Create an NWB file with a single camera having 3 video segments."""
    nwbfile = mock_NWBFile()
    for i in range(3):
        dst = tmp_path / f"cam1_seg{i}.mp4"
        shutil.copy(STUB_H264_PATH, dst)

    nwbfile.add_acquisition(
        ImageSeries(
            name="cam1",
            format="external",
            external_file=[f"./cam1_seg{i}.mp4" for i in range(3)],
            starting_frame=[0, 1000, 2000],
            starting_time=starting_time,
            rate=rate,
            unit="n.a.",
        )
    )
    return _save_and_load(nwbfile, tmp_path / "test.nwb")


# --- Duration parsing ---


class TestGetVideoDuration:
    def test_mp4_duration(self):
        assert _get_video_duration(STUB_H264_PATH) == pytest.approx(0.5, abs=0.05)


# --- get_videos unit tests ---


class TestGetVideosSingleFile:
    def test_returns_correct_local_time_at_start(self, nwbfile_with_single_video):
        result = get_videos(nwbfile_with_single_video, 0.0)

        assert "VideoCamera" in result
        path, local_time = result["VideoCamera"]
        assert local_time == pytest.approx(0.0)
        assert path.name == "test_video.mp4"

    def test_returns_correct_local_time_midway(self, nwbfile_with_single_video):
        result = get_videos(nwbfile_with_single_video, 0.2)

        assert "VideoCamera" in result
        _, local_time = result["VideoCamera"]
        assert local_time == pytest.approx(0.2)

    def test_past_end_omitted(self, nwbfile_with_single_video):
        # stub is 0.5s at 30fps = 15 frames; session_time 100s is way past the end
        result = get_videos(nwbfile_with_single_video, 100.0)
        assert "VideoCamera" not in result


class TestGetVideosMultiSegment:
    def test_first_segment(self, tmp_path):
        nwbfile = _make_multi_segment_nwbfile(tmp_path, rate=30.0)
        # Frame 150 → file 0 (starting_frame 0), local_frame 150, local_time 5.0s
        result = get_videos(nwbfile, 5.0)

        assert "cam1" in result
        path, local_time = result["cam1"]
        assert path.name == "cam1_seg0.mp4"
        assert local_time == pytest.approx(5.0)

    def test_second_segment(self, tmp_path):
        nwbfile = _make_multi_segment_nwbfile(tmp_path, rate=30.0)
        # session_time=35.0 → global_frame=1050 → file 1 (starts at 1000)
        # local_frame=50, local_time=50/30≈1.667
        result = get_videos(nwbfile, 35.0)

        assert "cam1" in result
        path, local_time = result["cam1"]
        assert path.name == "cam1_seg1.mp4"
        assert local_time == pytest.approx(50.0 / 30.0)

    def test_third_segment(self, tmp_path):
        nwbfile = _make_multi_segment_nwbfile(tmp_path, rate=30.0)
        # session_time=67.0 → global_frame=2010 → file 2 (starts at 2000)
        # local_frame=10, local_time=10/30≈0.333
        result = get_videos(nwbfile, 67.0)

        assert "cam1" in result
        path, local_time = result["cam1"]
        assert path.name == "cam1_seg2.mp4"
        assert local_time == pytest.approx(10.0 / 30.0)

    def test_exact_segment_boundary(self, tmp_path):
        nwbfile = _make_multi_segment_nwbfile(tmp_path, rate=30.0)
        # session_time=1000/30≈33.333 → global_frame=1000 → file 1 boundary
        session_time = 1000.0 / 30.0
        result = get_videos(nwbfile, session_time)

        assert "cam1" in result
        path, local_time = result["cam1"]
        assert path.name == "cam1_seg1.mp4"
        assert local_time == pytest.approx(0.0)


class TestGetVideosBeforeStart:
    def test_negative_session_time(self, nwbfile_with_single_video):
        result = get_videos(nwbfile_with_single_video, -1.0)
        assert len(result) == 0

    def test_before_starting_time(self, tmp_path):
        nwbfile = _make_multi_segment_nwbfile(tmp_path, rate=30.0, starting_time=10.0)
        result = get_videos(nwbfile, 5.0)
        assert "cam1" not in result


class TestGetVideosStartingFrameOffset:
    def test_offset_camera_omitted_before_offset(self, tmp_path):
        """Camera with starting_frame=[100] is omitted before frame 100."""
        nwbfile = mock_NWBFile()
        dst = tmp_path / "cam_offset.mp4"
        shutil.copy(STUB_H264_PATH, dst)

        nwbfile.add_acquisition(
            ImageSeries(
                name="cam_offset",
                format="external",
                external_file=["./cam_offset.mp4"],
                starting_frame=[100],
                starting_time=0.0,
                rate=120.0,
                unit="n.a.",
            )
        )
        nwbfile = _save_and_load(nwbfile, tmp_path / "test.nwb")

        # session_time=0.5 → global_frame=60 < starting_frame=100 → omitted
        result = get_videos(nwbfile, 0.5)
        assert "cam_offset" not in result

    def test_offset_camera_returns_correct_local_time(self, tmp_path):
        """Camera with starting_frame=[100] returns correct local_time after offset."""
        nwbfile = mock_NWBFile()
        dst = tmp_path / "cam_offset.mp4"
        shutil.copy(STUB_H264_PATH, dst)

        nwbfile.add_acquisition(
            ImageSeries(
                name="cam_offset",
                format="external",
                external_file=["./cam_offset.mp4"],
                starting_frame=[100],
                starting_time=0.0,
                rate=120.0,
                unit="n.a.",
            )
        )
        nwbfile = _save_and_load(nwbfile, tmp_path / "test.nwb")

        # session_time=1.0 → global_frame=120, file starts at 100
        # local_frame=20, local_time=20/120≈0.1667
        result = get_videos(nwbfile, 1.0)
        assert "cam_offset" in result
        _, local_time = result["cam_offset"]
        assert local_time == pytest.approx(20.0 / 120.0)


class TestGetVideosMultipleCameras:
    def test_both_cameras_returned(self, tmp_path):
        nwbfile = mock_NWBFile()
        for cam in ["cam1", "cam2"]:
            dst = tmp_path / f"{cam}.mp4"
            shutil.copy(STUB_H264_PATH, dst)
            nwbfile.add_acquisition(
                ImageSeries(
                    name=cam,
                    format="external",
                    external_file=[f"./{cam}.mp4"],
                    starting_frame=[0],
                    starting_time=0.0,
                    rate=30.0,
                    unit="n.a.",
                )
            )
        nwbfile = _save_and_load(nwbfile, tmp_path / "test.nwb")

        result = get_videos(nwbfile, 0.1)
        assert "cam1" in result
        assert "cam2" in result


# --- Integration tests with PAIR-R24M data ---


# QUICK FIX, for proper PR, downsample/trim videos: https://drive.google.com/drive/u/0/folders/1Qr8Kb6qzlDbiW10gqAj0x8yf_5CWaN5c
# and add in tests/fixtures
PAIR24_DIR = Path(r"C:\Users\aksel\Desktop\pair24")
PAIR24_NWB = PAIR24_DIR / "pair24.nwb"
PAIR24_CAM2_LONG_NWB = PAIR24_DIR / "pair24_cam2_long.nwb"

skip_no_pair24 = pytest.mark.skipif(not PAIR24_NWB.exists(), reason="PAIR-R24M data not available")


@skip_no_pair24
class TestGetVideosPair24Aligned:
    """Integration tests with pair24.nwb (both cameras aligned, 3 segments each at 120fps)."""

    @pytest.fixture(autouse=True)
    def load_nwb(self):
        self.nwbfile = read_nwb(str(PAIR24_NWB))

    def test_time_zero_first_segment(self):
        result = get_videos(self.nwbfile, 0.0)
        assert "cam1" in result
        assert "cam2" in result
        assert result["cam1"][0].name == "cam-1_frame-0to3499.mp4"
        assert result["cam2"][0].name == "cam-2_frame-0to3499.mp4"
        assert result["cam1"][1] == pytest.approx(0.0)
        assert result["cam2"][1] == pytest.approx(0.0)

    def test_second_segment(self):
        # session_time=30.0 → global_frame=3600 → file 1 (starts at 3500)
        result = get_videos(self.nwbfile, 30.0)
        assert result["cam1"][0].name == "cam-1_frame-3500to6999.mp4"
        assert result["cam1"][1] == pytest.approx(100.0 / 120.0)

    def test_third_segment(self):
        # session_time=60.0 → global_frame=7200 → file 2 (starts at 7000)
        result = get_videos(self.nwbfile, 60.0)
        assert result["cam1"][0].name == "cam-1_frame-7000to10499.mp4"
        assert result["cam1"][1] == pytest.approx(200.0 / 120.0)

    def test_past_end_omitted(self):
        # 10500 frames at 120fps = 87.5s; session_time=100 is past the end
        result = get_videos(self.nwbfile, 100.0)
        assert "cam1" not in result
        assert "cam2" not in result

    def test_segment_boundary(self):
        # Exact boundary at frame 3500 → session_time = 3500/120
        session_time = 3500.0 / 120.0
        result = get_videos(self.nwbfile, session_time)
        assert result["cam1"][0].name == "cam-1_frame-3500to6999.mp4"
        assert result["cam1"][1] == pytest.approx(0.0)


@skip_no_pair24
class TestGetVideosPair24Cam2Long:
    """Integration tests with pair24_cam2_long.nwb (cam2 has starting_frame=[100])."""

    @pytest.fixture(autouse=True)
    def load_nwb(self):
        self.nwbfile = read_nwb(str(PAIR24_CAM2_LONG_NWB))

    def test_time_zero_cam2_omitted(self):
        # cam2 starts at frame 100, session_time=0 → global_frame=0 < 100 → omitted
        result = get_videos(self.nwbfile, 0.0)
        assert "cam1" in result
        assert "cam2" not in result

    def test_cam2_appears_after_offset(self):
        # cam2 starts at frame 100 → session_time = 100/120 ≈ 0.833s
        # At session_time=1.0 → global_frame=120, local_frame=20, local_time≈0.167
        result = get_videos(self.nwbfile, 1.0)
        assert "cam1" in result
        assert "cam2" in result
        assert result["cam2"][0].name == "cam2_frame-100to10499.mp4"
        assert result["cam2"][1] == pytest.approx(20.0 / 120.0)

    def test_cam1_second_segment_cam2_still_first(self):
        # session_time=30.0: cam1 → file 1 (3500), cam2 → still file 0 (100)
        result = get_videos(self.nwbfile, 30.0)
        assert result["cam1"][0].name == "cam-1_frame-3500to6999.mp4"
        assert result["cam2"][0].name == "cam2_frame-100to10499.mp4"
        # cam2 local_time = (3600-100)/120 = 29.167
        assert result["cam2"][1] == pytest.approx(3500.0 / 120.0)


@skip_no_pair24
class TestGetVideoDurationPair24:
    def test_segment_duration(self):
        path = PAIR24_DIR / "cam-1_frame-0to3499.mp4"
        assert _get_video_duration(path) == pytest.approx(3500.0 / 120.0, abs=0.01)

    def test_concatenated_duration(self):
        path = PAIR24_DIR / "cam2_frame-100to10499.mp4"
        assert _get_video_duration(path) == pytest.approx(10400.0 / 120.0, abs=0.01)
