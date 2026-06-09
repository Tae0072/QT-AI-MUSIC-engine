<h1 align="center">QT-AI-MUSIC-engine</h1>

<p align="center">
  <b>가사 → 완성곡(보컬+반주) 생성 엔진 · YuE / YuEGP 기반 · Windows 친화 빌드</b><br>
  <i>Lyrics-to-song generation engine, based on YuE / YuEGP, with Windows-friendly tooling</i>
</p>

<p align="center">
  <a href="#한국어-안내">한국어</a> · <a href="#english-summary">English</a> ·
  <a href="https://github.com/deepbeepmeep/YuEGP">YuEGP (upstream)</a> ·
  <a href="https://github.com/multimodal-art-projection/YuE">YuE (원본)</a>
</p>

> ⚖️ **라이선스: CC BY-NC 4.0 (비상업)** — 원본 YuE/YuEGP의 조건을 그대로 따릅니다. 상업적 이용은 허용되지 않습니다. 자세한 내용은 [LICENSE](./LICENSE)와 [NOTICE](./NOTICE)를 확인하세요.

---

## 한국어 안내

### 소개

**QT-AI-MUSIC-engine**은 가사와 장르(분위기·악기·보컬 스타일)를 입력하면 **보컬과 반주가 포함된 완성곡**을 생성하는 음악 생성 엔진입니다. M-A-P/HKUST의 오픈 모델 **YuE**와, 적은 VRAM에서도 돌아가도록 최적화한 **YuEGP(YuE for the GPU Poor, by deepbeepmeep)**를 기반으로 하며, 이 저장소는 여기에 **Windows 실행 편의 기능**(실행/종료 스크립트, 로딩바, 실행 오류 수정)과 **한국어 문서**를 더한 파생 버전입니다.

- 가사 + 장르 프롬프트만으로 수 분 길이의 곡 생성
- 다양한 장르·보컬 스타일 지원, 오디오 프롬프트(ICL)로 스타일 유도 가능
- VRAM 사정에 맞춘 성능 프로파일(16GB → 10GB 미만까지)
- Windows에서 더블클릭 실행을 돕는 보조 스크립트 포함

### 동작 원리(요약)

2단계 파이프라인입니다. **Stage 1**에서 가사·장르를 받아 곡의 토큰을 생성하고, **Stage 2**에서 오디오 코덱(`xcodec_mini_infer`)으로 실제 음원을 디코딩합니다.

### 요구 사항

- **Python 3.10** 권장 (3.11 가능, 3.12/3.13은 이슈 보고됨)
- **NVIDIA GPU** — 프로파일에 따라 VRAM 약 10~16GB. AMD(ROCm)도 일부 지원
- **git** + **git-lfs** (모델 다운로드용)
- CUDA 12.4 환경의 PyTorch 2.5.1

> 참고: 모델 가중치(`models/`, `inference/xcodec_mini_infer/`)와 가상환경(`venv310/`)은 용량이 커서 저장소에 포함되지 않습니다. 아래 절차대로 별도로 받습니다.

### 설치

```bash
# 1) 소스 코드 받기 (git-lfs 필요)
git lfs install
git clone https://github.com/Tae0072/QT-AI-MUSIC-engine.git
cd QT-AI-MUSIC-engine

# 2) 오디오 코덱 모델 받기
cd inference
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
cd ..

# 3) 가상환경 + PyTorch (CUDA 12.4)
python -m venv venv310
# Windows:  venv310\Scripts\activate
# Linux:    source venv310/bin/activate
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124

# 4) 의존성 설치
pip install -r requirements.txt
```

AMD GPU(ROCm) 사용 시:

```bash
pip3 install torch torchaudio triton --index-url https://download.pytorch.org/whl/rocm6.2
```

(선택) 저VRAM 가속 패치 — 프로파일 4/5 또는 16GB+ 가속용:

```bash
# Windows
patchtransformers.bat
# Linux
source patchtransformers.sh
```

(선택) FlashAttention 2 — 메모리 절약/속도 향상:

```bash
pip install flash-attn --no-build-isolation
```

> FlashAttention 설치가 어려우면(특히 Windows) 실행 시 `--sdpa` 옵션으로 대체할 수 있습니다(VRAM은 더 쓸 수 있음).

### 실행

가장 쉬운 방법은 **Gradio 웹 UI**입니다.

```bash
cd inference
python gradio_server.py            # 가사 + 장르 모드(기본)
python gradio_server.py --icl      # 오디오 프롬프트(ICL) 모드
```

**성능 프로파일** (VRAM에 맞춰 선택):

| 프로파일 | 특징 | 권장 VRAM |
|:---:|---|:---:|
| `--profile 1` | 가장 빠름 | 16GB+ |
| `--profile 3` | 8비트 양자화, 조금 느림 | 12GB+ |
| `--profile 4` | 순차 오프로딩, 느림 | 10GB 미만 |
| `--profile 5` | 최소 VRAM | 최저 |

```bash
cd inference
python gradio_server.py --profile 3
```

추가 옵션: `--compile`(Triton 필요, 가속), `--sdpa`(FlashAttention 대체), `--turbo-stage2`(Stage 2 약 2배 가속, 16GB+).

VRAM이 충분한데 OOM이 나면 실행 전:

```bash
# Linux/WSL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

**CLI 직접 실행**(`infer.py`)도 가능합니다:

```bash
cd inference
python infer.py \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt prompt_examples/genre.txt \
    --lyrics_txt prompt_examples/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ./output \
    --cuda_idx 0 \
    --max_new_tokens 3000
```

### Windows 보조 스크립트

이 저장소에는 Windows에서 더블클릭으로 쓰기 편하도록 보조 스크립트가 포함돼 있습니다.

- `YuE_실행.vbs` — 앱 실행
- `YuE_종료.bat` — 실행 중인 프로세스 종료
- `YuE_로딩바.ps1` — 진행 상황 로딩바 표시

### 프롬프트 작성 팁

- **장르 태그**는 5요소를 권장: 장르 / 악기 / 분위기 / 성별 / 음색. 음색에는 "vocal"을 포함(예: `bright vocal`).
  예) `inspiring female uplifting pop airy vocal electronic bright vocal vocal`
- 자주 쓰는 태그 200종은 [`wav_top_200_tags.json`](./wav_top_200_tags.json) 참고.
- 가사 길이와 `--max_new_tokens`를 맞추세요(3000 ≈ 약 30초/세그먼트).
- 오디오 프롬프트는 약 30초 분량이 적당합니다.

### 더 알아보기

저장소의 한국어 문서도 참고하세요: 입문용 스터디 노트(`스터디노트_YuE_입문.md`), 작업 워크플로우(`워크플로우_YuE_작업.md`), 실행 오류 수정·로딩바 작업 리포트(`작업리포트_실행오류수정_및_로딩바.md`).

---

## English Summary

**QT-AI-MUSIC-engine** turns lyrics + genre prompts into complete songs (vocals + accompaniment). It is built on **YuE** (M-A-P / HKUST) and **YuEGP** ("YuE for the GPU Poor", by deepbeepmeep), adding **Windows-friendly launch tooling** (run/stop scripts, a loading bar, execution-error fixes) and **Korean documentation**.

**Quick start**

```bash
git lfs install
git clone https://github.com/Tae0072/QT-AI-MUSIC-engine.git
cd QT-AI-MUSIC-engine/inference
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
cd ..
python -m venv venv310 && venv310\Scripts\activate   # Windows
pip install torch==2.5.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/test/cu124
pip install -r requirements.txt
cd inference && python gradio_server.py --profile 3
```

Python 3.10 recommended. Choose a `--profile` (1=16GB+, 3=12GB, 4/5=low VRAM). Model weights and the virtualenv are excluded from the repo and downloaded separately.

**License:** CC BY-NC 4.0 (non-commercial), following the upstream YuE/YuEGP terms. See [LICENSE](./LICENSE) and [NOTICE](./NOTICE).

---

## 출처 및 크레딧 / Credits

- **YuEGP** — deepbeepmeep · https://github.com/deepbeepmeep/YuEGP
- **YuE** — M-A-P / HKUST · https://github.com/multimodal-art-projection/YuE
- **xcodec_mini_infer** — https://huggingface.co/m-a-p/xcodec_mini_infer

원본 YuE 논문 인용:

```BibTeX
@misc{yuan2025yue,
  title={YuE: Open Music Foundation Models for Full-Song Generation},
  author={Ruibin Yuan and Hanfeng Lin and Shawn Guo and Ge Zhang and Jiahao Pan and others},
  howpublished={\url{https://github.com/multimodal-art-projection/YuE}},
  year={2025},
  note={GitHub repository}
}
```

## License

This project is licensed under **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)**. Commercial use is not permitted. See [LICENSE](./LICENSE).
