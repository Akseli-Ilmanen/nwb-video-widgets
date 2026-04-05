"""Shared utilities for NWB video widgets."""

import socket
import struct
import threading
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from unittest import result

from pynwb import NWBFile
from pynwb.image import ImageSeries

# Global registry for video file servers
_video_servers: dict[str, tuple[HTTPServer, int]] = {}

# Codecs natively supported by all major browsers via HTML5 <video>
BROWSER_COMPATIBLE_CODECS = {"h264", "H264", "avc1", "vp8", "vp9", "VP8", "VP9", "vp09", "av01", "AV01"}

_HEADER_READ_SIZE = 32 * 1024  # 32 KB is enough for codec detection


def _detect_avi_codec(data: bytes) -> str | None:
    """Extract the video codec FourCC from AVI (RIFF) header bytes.

    Walks the RIFF chunk structure to find the ``strh`` chunk with
    ``fccType == b'vids'`` and returns the ``fccHandler`` field.
    """
    if len(data) < 12 or data[:4] != b"RIFF" or data[8:12] != b"AVI ":
        return None

    pos = 12
    while pos + 8 <= len(data):
        chunk_id = data[pos : pos + 4]
        if len(data) < pos + 8:
            break
        chunk_size = struct.unpack_from("<I", data, pos + 4)[0]

        if chunk_id == b"LIST":
            pos += 12  # enter LIST, skip list type
            continue

        if chunk_id == b"strh" and chunk_size >= 8:
            fcc_type = data[pos + 8 : pos + 12]
            fcc_handler = data[pos + 12 : pos + 16]
            if fcc_type == b"vids":
                codec = fcc_handler.decode("ascii", errors="replace").strip("\x00")
                return codec if codec else None

        pos += 8 + chunk_size + (chunk_size % 2)

    return None


def _find_mp4_box(data: bytes, start: int, end: int, target: bytes) -> tuple[int, int] | None:
    """Find an ISO BMFF box by type within a byte range.

    Returns ``(payload_start, payload_end)`` or ``None``.
    """
    pos = start
    while pos + 8 <= end:
        box_size = struct.unpack_from(">I", data, pos)[0]
        box_type = data[pos + 4 : pos + 8]

        if box_size == 1 and pos + 16 <= end:
            box_size = struct.unpack_from(">Q", data, pos + 8)[0]
            payload_start = pos + 16
        elif box_size < 8:
            break
        else:
            payload_start = pos + 8

        if box_type == target:
            return payload_start, min(pos + box_size, end)

        pos += box_size

    return None


def _detect_mp4_codec(data: bytes) -> str | None:
    """Extract the video codec FourCC from MP4/MOV header bytes.

    Navigates ``moov > trak > mdia > minf > stbl > stsd`` and reads the
    codec identifier from the first sample entry.  If the ``moov`` box
    is not found at a top-level box boundary (e.g. when parsing the tail
    of a file), falls back to scanning for the ``moov`` signature.
    """
    end = len(data)
    inner_path = [b"trak", b"mdia", b"minf", b"stbl", b"stsd"]

    # Try structured traversal first
    moov = _find_mp4_box(data, 0, end, b"moov")

    # Fallback: scan for moov signature (useful for tail-of-file reads)
    if moov is None:
        search_pos = 0
        while True:
            found = data.find(b"moov", search_pos)
            if found == -1 or found < 4:
                return None
            box_size = struct.unpack_from(">I", data, found - 4)[0]
            if 16 < box_size <= end - (found - 4):
                moov = (found + 4, found - 4 + box_size)
                break
            search_pos = found + 4

    start, end = moov
    for box_type in inner_path:
        result = _find_mp4_box(data, start, end, box_type)
        if result is None:
            return None
        start, end = result

    # stsd FullBox: version(1) + flags(3) + entry_count(4) = 8 bytes
    # then SampleEntry: size(4) + codec_fourcc(4)
    entry_offset = start + 8
    if entry_offset + 8 > end:
        return None

    codec_fourcc = data[entry_offset + 4 : entry_offset + 8]
    return codec_fourcc.decode("ascii", errors="replace").strip("\x00") or None


def detect_video_codec(video_path: Path) -> str | None:
    """Detect the video codec of a file by reading its header bytes.

    Supports AVI (RIFF) and MP4/MOV (ISO BMFF) containers. Returns the
    codec identifier string (e.g. ``"avc1"``, ``"MJPG"``, ``"mp4v"``)
    or ``None`` if the format is not recognized.

    For MP4 files where the ``moov`` box is at the end of the file (common
    when not encoded with ``faststart``), the tail of the file is also read.

    Parameters
    ----------
    video_path : Path
        Path to a video file.

    Returns
    -------
    str or None
        Codec identifier, or None if unrecognized.
    """
    file_size = video_path.stat().st_size
    with open(video_path, "rb") as f:
        data = f.read(_HEADER_READ_SIZE)

    if len(data) < 12:
        return None

    # AVI: RIFF....AVI
    if data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return _detect_avi_codec(data)

    # MP4/MOV: ftyp box at start
    if data[4:8] == b"ftyp" or data[4:8] == b"moov":
        codec = _detect_mp4_codec(data)
        if codec is not None:
            return codec

        # moov may be at the end of the file (no faststart)
        if file_size > _HEADER_READ_SIZE:
            tail_size = min(file_size, _HEADER_READ_SIZE * 8)  # up to 256KB
            with open(video_path, "rb") as f:
                f.seek(file_size - tail_size)
                tail_data = f.read(tail_size)
            return _detect_mp4_codec(tail_data)

    return None


def validate_video_codec(video_path: Path) -> None:
    """Raise ``ValueError`` if the video uses a non-browser-compatible codec.

    Parameters
    ----------
    video_path : Path
        Path to a video file.

    Raises
    ------
    ValueError
        If the detected codec is not in ``BROWSER_COMPATIBLE_CODECS``.
    """
    codec = detect_video_codec(video_path)
    if codec is None:
        return  # unrecognized format, don't block

    if codec not in BROWSER_COMPATIBLE_CODECS:
        stem = video_path.stem
        raise ValueError(
            f"Video '{video_path.name}' uses the '{codec}' codec which cannot be played in the browser. "
            f"Re-encode with: ffmpeg -i {video_path.name} -c:v libx264 -crf 18 -pix_fmt yuv420p {stem}_h264.mp4"
        )


def discover_video_series(nwbfile: NWBFile) -> dict[str, ImageSeries]:
    """Discover all ImageSeries with external video files in an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to search for video series

    Returns
    -------
    dict[str, ImageSeries]
        Mapping of series names to ImageSeries objects that have external_file
    """
    video_series = {}
    for name, obj in nwbfile.acquisition.items():
        if isinstance(obj, ImageSeries) and obj.external_file is not None:
            video_series[name] = obj
    return video_series


def get_video_timestamps(nwbfile: NWBFile) -> dict[str, list[float]]:
    """Extract video timestamps from all ImageSeries in an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing video ImageSeries in acquisition

    Returns
    -------
    dict[str, list[float]]
        Mapping of video names to timestamp arrays
    """
    video_series = discover_video_series(nwbfile)
    timestamps = {}

    for name, series in video_series.items():
        if series.timestamps is not None:
            timestamps[name] = [float(t) for t in series.timestamps[:]]
        elif series.starting_time is not None:
            timestamps[name] = [float(series.starting_time)]
        else:
            timestamps[name] = [0.0]

    return timestamps


def get_video_info(nwbfile: NWBFile) -> dict[str, dict]:
    """Extract video time range information from all ImageSeries in an NWB file.

    Uses indexed access (timestamps[0], timestamps[-1]) instead of loading
    the full timestamps array, which is important for DANDI streaming where
    each slice triggers HTTP range requests.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing video ImageSeries in acquisition

    Returns
    -------
    dict[str, dict]
        Mapping of video names to info dictionaries with keys:
        - start: float, start time in seconds
        - end: float, end time in seconds
    """
    video_series = discover_video_series(nwbfile)
    info = {}

    for name, series in video_series.items():
        if series.timestamps is not None and len(series.timestamps) > 0:
            start = float(series.timestamps[0])
            end = float(series.timestamps[-1])
        elif series.starting_time is not None:
            start = float(series.starting_time)
            # Without timestamps, we can't determine end time accurately
            # Use starting_time as both start and end
            end = start
        else:
            start = 0.0
            end = 0.0

        info[name] = {
            "start": start,
            "end": end,
        }

    return info


def _get_mp4_duration_from_data(data: bytes) -> float | None:
    """Extract movie duration in seconds from MP4/MOV data via the ``mvhd`` box."""
    end = len(data)
    moov = _find_mp4_box(data, 0, end, b"moov")

    if moov is None:
        search_pos = 0
        while True:
            found = data.find(b"moov", search_pos)
            if found == -1 or found < 4:
                return None
            box_size = struct.unpack_from(">I", data, found - 4)[0]
            if 16 < box_size <= end - (found - 4):
                moov = (found + 4, found - 4 + box_size)
                break
            search_pos = found + 4

    if moov is None:
        return None

    start, end = moov
    mvhd = _find_mp4_box(data, start, end, b"mvhd")
    if mvhd is None:
        return None

    start, end = mvhd
    version = data[start]
    if version == 1:
        if start + 32 > end:
            return None
        timescale = struct.unpack_from(">I", data, start + 20)[0]
        duration = struct.unpack_from(">Q", data, start + 24)[0]
    else:
        if start + 20 > end:
            return None
        timescale = struct.unpack_from(">I", data, start + 12)[0]
        duration = struct.unpack_from(">I", data, start + 16)[0]

    if timescale == 0:
        return None
    return duration / timescale


def _get_video_duration(video_path: Path) -> float | None:
    """Get duration of a video file in seconds by parsing MP4/MOV container metadata.

    Returns ``None`` if the format is not recognized or the duration
    cannot be determined.
    """
    file_size = video_path.stat().st_size
    with open(video_path, "rb") as f:
        data = f.read(_HEADER_READ_SIZE)

    if len(data) < 12:
        return None

    if data[4:8] in (b"ftyp", b"moov"):
        duration = _get_mp4_duration_from_data(data)
        if duration is not None:
            return duration
        if file_size > _HEADER_READ_SIZE:
            tail_size = min(file_size, _HEADER_READ_SIZE * 8)
            with open(video_path, "rb") as f:
                f.seek(file_size - tail_size)
                tail_data = f.read(tail_size)
            return _get_mp4_duration_from_data(tail_data)

    return None


def _resolve_video_path(external_path: str, base_dir: Path | None) -> Path:
    """Resolve an external_file path to a Path, relative to the NWB file directory."""
    path = Path(external_path)
    if path.is_absolute():
        return path
    if base_dir is not None:
        return (base_dir / path).resolve()
    return path


def get_videos(nwbfile: NWBFile, session_time: float) -> dict[str, tuple[Path, float]]:       
    """Map a session timestamp to the corresponding video file and local time for each camera.                                                                                                                                                                                                   
    For each ``ImageSeries`` with external video files in the NWB file,
    determines which video file contains the given session time and computes the
    local time within that file.

    Supports two NWB timing schemes:
    - **Rate mode**: ``rate`` + ``starting_time`` — evenly spaced frames.
    - **Timestamps mode**: explicit ``timestamps`` array — one per frame,
    supports gaps between files (e.g. per-trial videos).

    Parameters
    ----------
    nwbfile : NWBFile
        An open NWBFile containing ImageSeries acquisitions with external video
        files.  If loaded from disk the file paths are resolved relative to the
        NWB file location; absolute ``external_file`` paths work regardless.
    session_time : float
        Time in seconds from session start.

    Returns
    -------
    dict[str, tuple[Path, float]]
        Mapping of acquisition name to ``(video_file_path, local_time_within_file)``.
        Streams where ``session_time`` falls outside the recorded range are
        omitted from the result.

    Examples
    --------
    >>> result = get_videos(nwbfile, session_time=35.0)
    >>> result
    {"video_cam-1": (Path("cam1-video1.mp4"), 5.0)}
    """
    video_series = discover_video_series(nwbfile)
    
    base_dir = None
    if nwbfile.read_io is not None and hasattr(nwbfile.read_io, "source"):
        base_dir = Path(nwbfile.read_io.source).parent

    result: dict[str, tuple[Path, float]] = {}
    for name, series in video_series.items():
        files = list(series.external_file)
        starting_frames = (
            [int(f) for f in series.starting_frame]
            if series.starting_frame is not None
            else [0] * len(files)
        )
        if len(starting_frames) != len(files):
            continue

        rate = series.rate
        timestamps = getattr(series, "timestamps", None)

        if timestamps is not None and len(timestamps) > 0:
            # Timestamps mode: searchsorted into the timestamps array
            ts = np.asarray(timestamps)
            if session_time < ts[0] or session_time > ts[-1]:
                continue

            global_idx = int(np.searchsorted(ts, session_time, side="right")) - 1
            global_idx = max(0, min(global_idx, len(ts) - 1))

            # Find which file this index belongs to
            file_idx = np.searchsorted(starting_frames, global_idx, side="right") - 1
            file_idx = max(0, min(file_idx, len(files) - 1))

            local_idx = global_idx - starting_frames[file_idx]
            # Local time = time elapsed since this file's first frame
            file_first_ts = ts[starting_frames[file_idx]]
            local_time = float(ts[global_idx]) - float(file_first_ts)

            result[name] = (_resolve_video_path(files[file_idx], base_dir), local_time)

        elif rate is not None and rate > 0:
            # Rate mode: evenly spaced frames
            starting_time = float(series.starting_time) if series.starting_time is not None else 0.0
            global_frame = (session_time - starting_time) * rate
            if global_frame < 0:
                continue

            if global_frame < starting_frames[0]:
                continue

            for i in range(len(files)):
                file_start = starting_frames[i]
                if i + 1 < len(files):
                    file_end = float(starting_frames[i + 1])
                else:
                    file_path = _resolve_video_path(files[i], base_dir)
                    if file_path.is_file():
                        duration = _get_video_duration(file_path)
                        file_end = file_start + duration * rate if duration is not None else float("inf")
                    else:
                        file_end = float("inf")

                if file_start <= global_frame < file_end:
                    local_time = (global_frame - file_start) / rate
                    result[name] = (_resolve_video_path(files[i], base_dir), local_time)
                    break

    return result


class _RangeRequestHandler(SimpleHTTPRequestHandler):
    """HTTP request handler with CORS headers and Range request support for video streaming."""

    def send_head(self):
        """Handle HEAD requests and Range requests for partial content."""
        path = self.translate_path(self.path)

        if not Path(path).is_file():
            return super().send_head()

        file_size = Path(path).stat().st_size
        range_header = self.headers.get("Range")

        if range_header:
            # Parse Range header (e.g., "bytes=0-1023")
            try:
                range_spec = range_header.replace("bytes=", "")
                start_str, end_str = range_spec.split("-")
                start = int(start_str) if start_str else 0
                end = int(end_str) if end_str else file_size - 1
                end = min(end, file_size - 1)
                content_length = end - start + 1

                f = open(path, "rb")
                f.seek(start)

                self.send_response(206)  # Partial Content
                self.send_header("Content-Type", self.guess_type(path))
                self.send_header("Content-Length", str(content_length))
                self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
                self.send_header("Access-Control-Allow-Headers", "Range")
                self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
                self.end_headers()
                return f
            except (ValueError, IOError):
                pass

        # No Range header or invalid range - serve full file
        return super().send_head()

    def end_headers(self):
        """Add CORS headers to all responses."""
        # Only add if not already added (for non-range requests)
        if not self._headers_buffer or b"Access-Control-Allow-Origin" not in b"".join(self._headers_buffer):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
            self.send_header("Access-Control-Allow-Headers", "Range")
            self.send_header("Accept-Ranges", "bytes")
        super().end_headers()

    def do_OPTIONS(self):
        """Handle CORS preflight requests."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress logging to avoid cluttering notebook output."""
        pass

    def handle(self):
        """Handle requests, suppressing connection reset errors."""
        try:
            super().handle()
        except (ConnectionResetError, BrokenPipeError):
            # Browser closed connection early - this is normal during video seeking
            pass


def _find_free_port() -> int:
    """Find a free port on localhost."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def start_video_server(directory: Path) -> int:
    """Start an HTTP server to serve video files from a directory.

    If a server is already running for this directory, returns its port.

    Parameters
    ----------
    directory : Path
        Directory containing video files to serve

    Returns
    -------
    int
        Port number the server is listening on
    """
    dir_key = str(directory.resolve())

    # Return existing server port if already running
    if dir_key in _video_servers:
        _, port = _video_servers[dir_key]
        return port

    port = _find_free_port()
    handler = partial(_RangeRequestHandler, directory=str(directory))
    server = HTTPServer(("127.0.0.1", port), handler)

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    _video_servers[dir_key] = (server, port)
    return port


def discover_pose_estimation_cameras(nwbfile: NWBFile) -> dict:
    """Discover all PoseEstimation containers in an NWB file.

    Searches all objects in the file regardless of where they are stored,
    so PoseEstimation data in any processing module (e.g. 'pose_estimation',
    'behavior') is found.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file to search for pose estimation data

    Returns
    -------
    dict
        Mapping of camera names to PoseEstimation objects.
    """
    cameras = {}
    for obj in nwbfile.objects.values():
        if obj.neurodata_type == "PoseEstimation":
            assert obj.name not in cameras, f"Duplicate PoseEstimation name found: {obj.name}"
            cameras[obj.name] = obj
    return cameras


def get_camera_to_video_mapping(nwbfile: NWBFile) -> dict[str, str]:
    """Auto-map pose estimation camera names to video series names.

    Uses the naming convention: camera name prefixed with "Video"
    - 'LeftCamera' -> 'VideoLeftCamera'
    - 'BodyCamera' -> 'VideoBodyCamera'

    Only returns mappings where both the camera and corresponding video exist.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing pose estimation and video data

    Returns
    -------
    dict[str, str]
        Mapping from camera names to video series names
    """
    cameras = discover_pose_estimation_cameras(nwbfile)
    video_series = discover_video_series(nwbfile)

    mapping = {}
    for camera_name in cameras:
        video_name = f"Video{camera_name}"
        if video_name in video_series:
            mapping[camera_name] = video_name

    return mapping


def get_pose_estimation_info(nwbfile: NWBFile) -> dict[str, dict]:
    """Extract pose estimation info for all cameras in an NWB file.

    Parameters
    ----------
    nwbfile : NWBFile
        NWB file containing pose estimation in processing['pose_estimation']

    Returns
    -------
    dict[str, dict]
        Mapping of camera names to info dictionaries with keys:
        - start: float, start time in seconds
        - end: float, end time in seconds
        - keypoints: list[str], names of keypoints
    """
    cameras = discover_pose_estimation_cameras(nwbfile)
    info = {}

    for camera_name, pose_estimation in cameras.items():
        # Get keypoint names (remove PoseEstimationSeries suffix)
        keypoint_names = [
            name.replace("PoseEstimationSeries", "") for name in pose_estimation.pose_estimation_series.keys()
        ]

        # Get start/end times from the first pose estimation series using indexed
        # access to avoid loading the full timestamps array into memory. This is
        # important for DANDI streaming where each slice triggers HTTP range requests.
        first_series = next(iter(pose_estimation.pose_estimation_series.values()), None)
        if first_series is not None and first_series.timestamps is not None:
            start = float(first_series.timestamps[0])
            end = float(first_series.timestamps[-1])
        else:
            start = 0.0
            end = 0.0

        info[camera_name] = {
            "start": start,
            "end": end,
            "keypoints": keypoint_names,
        }

    return info
