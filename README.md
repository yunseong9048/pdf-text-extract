# v101 PDF 원문 텍스트 추출

PDF 텍스트층을 읽어 페이지·단 읽기 순서를 정리하고 TXT와 JSON으로 저장합니다. 표와 수식은 구조를 안전하게 복원할 수 있을 때 보기 쉬운 텍스트로 출력하고, 그림 안의 축·범례 텍스트는 본문에서 분리합니다. 모든 판단은 페이지 기하와 텍스트 모양을 기준으로 하며 논문 ID별 분기는 사용하지 않습니다.

## 실행

Python 3.10 이상과 PyMuPDF가 필요합니다.

```bash
python3 -m pip install pymupdf
python3 v101_pdf_text_extract/extract_pdf_assets.py --limit 10
```

기본 입력은 `CatalystHub Zr precursor papers_all/Paper`, 기본 출력은 `outputs/v101_pdf_text_extract`입니다. 다른 폴더도 지정할 수 있습니다.

```bash
python3 v101_pdf_text_extract/extract_pdf_assets.py --input-dir "/path/to/pdfs" --output-dir "/path/to/output"
```

## 공통 처리

- PDF 글자 좌표, 기준선, 글꼴 크기를 이용해 단과 문단 순서를 정리하고 화학식의 위·아래 첨자를 복원합니다.
- 표 캡션과 표 영역을 찾고 PyMuPDF가 반환한 행·열이 충분히 일관될 때 Markdown 표로 출력합니다. 불확실하면 원문 블록과 crop을 보존하고 확인 표시를 남깁니다.
- 한 줄 또는 단순 다중행 수식은 글자 위치로 복원합니다. 분수선이 있거나 레이아웃이 모호하면 추측하지 않고 원문 crop과 검토 표시를 남깁니다.
- Figure/Fig./Scheme 캡션 근처의 그래픽 영역과 겹치는 짧은 축·눈금·범례 텍스트는 본문에서 제외합니다. 제외한 원문 문자열과 좌표는 JSON 경고에 보존합니다.
- PDF 내장 이미지, 그림 crop, 전체 페이지 렌더를 함께 저장합니다.
- 합자와 PDF에서 자주 쓰이는 결합 기호 코드를 정규화합니다. 알 수 없는 글리프는 JSON 경고에 남깁니다.

스캔 페이지는 OCR하지 않습니다. 표의 복잡한 병합 셀, 분수 수식, 그래프와 본문이 겹치는 레이아웃은 자동 판단이 제한되므로 JSON 경고와 crop을 원본 PDF와 대조해야 합니다.

## 출력

- `texts/*.txt`: 페이지 구분과 Markdown 표, 복원 가능한 수식이 포함된 본문
- `json/*.json`: 모든 원문 블록, 좌표, 표 후보 셀, 수식 원문/복원문, 제외된 그림 텍스트, 경고
- `pages/*/*.png`: 전체 페이지 렌더
- `images/*/*`: 표·수식·그림 확인용 crop 및 추출 이미지
- `manifest.json`, `summary.csv`: 처리 내역과 통계
- `quality_report.json`: 무결성 실패와 검수 경고
- `comparison.html`: PDF와 텍스트 대조 화면

`--self-test`는 내장 회귀 검사를, `--verify-only`는 기존 출력 검증을 실행합니다.
