import os
import tempfile
import time
from pathlib import Path
from datetime import timedelta

import streamlit as st
from faster_whisper import WhisperModel

st.set_page_config(
    page_title="Trích xuất phụ đề từ Video",
    page_icon="🎬",
    layout="wide",
)

# ============================================================
# CẤU HÌNH CHO MÔI TRƯỜNG WEB (CPU 2GB RAM)
# ============================================================
# Trên Streamlit Cloud / HF Spaces free tier:
# - CPU ~2 vCPU
# - RAM ~2-16GB (tùy platform)
# → Chỉ nên cho phép tiny/base/small

ALLOWED_MODELS = ["tiny", "base", "small"]


def format_timestamp(seconds: float) -> str:
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours = total_seconds // 3600
    minutes = (total_seconds % 3600) // 60
    secs = total_seconds % 60
    millis = int((seconds - total_seconds) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def segments_to_srt(segments) -> str:
    lines = []
    for i, seg in enumerate(segments, start=1):
        lines.append(str(i))
        lines.append(f"{format_timestamp(seg.start)} --> {format_timestamp(seg.end)}")
        lines.append(seg.text.strip())
        lines.append("")
    return "\n".join(lines)


@st.cache_resource(show_spinner=False)
def load_model(model_size: str):
    """Load model CPU int8 - tối ưu cho cloud free tier."""
    return WhisperModel(
        model_size,
        device="cpu",
        compute_type="int8",
        cpu_threads=2,        # giới hạn để không vượt quota
        num_workers=1,
    )


def transcribe_video(video_path: str, model_size: str, language: str, progress_cb=None):
    model = load_model(model_size)
    lang = None if language == "auto" else language

    segments_gen, info = model.transcribe(
        video_path,
        language=lang,
        beam_size=1,          # beam_size=1 nhanh hơn nhiều, đủ tốt cho web
        best_of=1,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500),
        condition_on_previous_text=False,  # tránh loop
    )

    total_duration = info.duration or 0
    segments = []
    for seg in segments_gen:
        segments.append(seg)
        if progress_cb and total_duration > 0:
            progress_cb(min(seg.end / total_duration, 1.0))
    if progress_cb:
        progress_cb(1.0)
    return segments, info


# ============================================================
# GIAO DIỆN
# ============================================================
st.title("🎬 Trích xuất phụ đề (.srt) từ Video")
st.caption("AI Whisper (faster-whisper) • Chạy hoàn toàn trên web")

with st.sidebar:
    st.header("⚙️ Cấu hình")

    model_size = st.selectbox(
        "Model AI",
        options=ALLOWED_MODELS,
        index=1,
        help="Web free tier chỉ chạy tốt tới `small`.",
    )

    language = st.selectbox(
        "Ngôn ngữ",
        options=["auto", "vi", "en", "zh", "ja", "ko"],
        format_func=lambda x: {
            "auto": "🌐 Tự động phát hiện",
            "vi": "🇻🇳 Tiếng Việt",
            "en": "🇬🇧 Tiếng Anh",
            "zh": "🇨🇳 Tiếng Trung",
            "ja": "🇯🇵 Tiếng Nhật",
            "ko": "🇰🇷 Tiếng Hàn",
        }[x],
        index=0,
    )

    st.markdown("---")
    st.warning(
        "⚠️ **Giới hạn web free tier:**\n"
        "- Video nên **< 10 phút**\n"
        "- Model tối đa: **small**\n"
        "- Nếu timeout, thử `tiny`/`base`"
    )

uploaded_file = st.file_uploader(
    "📤 Kéo & thả video (MP4, MKV, AVI, MOV, WEBM)",
    type=["mp4", "mkv", "avi", "mov", "webm", "flv", "wmv", "m4v"],
)

if uploaded_file is not None:
    st.video(uploaded_file)
    st.success(f"Đã tải lên: **{uploaded_file.name}** "
               f"({uploaded_file.size / 1024 / 1024:.1f} MB)")

    if st.button("🚀 Tạo phụ đề (.srt)", type="primary", use_container_width=True):
        suffix = Path(uploaded_file.name).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        progress_bar = st.progress(0.0, text="⏳ Đang khởi tạo model...")

        try:
            start = time.time()

            def cb(p):
                progress_bar.progress(p, text=f"🎙️ Đang nhận dạng... {p*100:.1f}%")

            with st.spinner("Đang xử lý (có thể mất vài phút)..."):
                segments, info = transcribe_video(
                    tmp_path, model_size, language, progress_cb=cb
                )

            elapsed = time.time() - start
            progress_bar.progress(1.0, text="✅ Hoàn tất!")

            st.info(
                f"🌐 Ngôn ngữ: **{info.language}** "
                f"({info.language_probability:.2%}) • "
                f"⏱️ {elapsed:.1f}s • 📝 {len(segments)} đoạn"
            )

            srt_content = segments_to_srt(segments)
            srt_filename = Path(uploaded_file.name).stem + ".srt"

            st.subheader("👀 Xem trước phụ đề")
            st.text_area("SRT", value=srt_content, height=350,
                         label_visibility="collapsed")

            st.download_button(
                "⬇️ Tải file .srt về máy",
                data=srt_content.encode("utf-8"),
                file_name=srt_filename,
                mime="application/x-subrip",
                type="primary",
                use_container_width=True,
            )

        except Exception as e:
            st.error(f"❌ Lỗi: {e}")
            st.exception(e)
        finally:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
else:
    st.info("👆 Tải lên một file video để bắt đầu.")

st.markdown("---")
st.caption("Made with ❤️ using Streamlit + faster-whisper")