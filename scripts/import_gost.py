# /// script
# requires-python = ">=3.12"
# dependencies = ["beautifulsoup4==4.14.3", "markdownify==1.2.2", "requests>=2.32,<3"]
# ///
"""Сохранить тексты стандартов и иллюстрации для автономного использования.

Запуск: uv run scripts/import_gost.py
HTML и изображения кешируются. URL и контрольные суммы записываются в manifest.json.
В индекс попадает Markdown; исходная верстка с рисунками доступна в offline/*.html.
"""

import json
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from markdownify import markdownify
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "data/sources/gost"
KNOWLEDGE = ROOT / "data/knowledge/standards"
MEGA = "https://meganorm.ru/mega_doc/norm/gost-r_gosudarstvennyj-standart/0/"
DOCUMENTS = [
    ("gost_r_2_109_2023", "ГОСТ Р 2.109-2023. Основные требования к чертежам",
     MEGA + "gost_r_2_109-2023_natsionalnyy_standart_rossiyskoy.html", 40000, "10.5.4"),
    ("gost_r_2_104_2023", "ГОСТ Р 2.104-2023. Основные надписи",
     MEGA + "gost_r_2_104-2023_natsionalnyy_standart_rossiyskoy.html", 15000, "Приложение Г"),
    ("gost_2_305_2008", "ГОСТ 2.305-2008. Изображения — виды, разрезы, сечения",
     "https://files.stroyinf.ru/Data2/1/4293830/4293830650.htm", 20000, "9.15"),
    ("gost_2_307_2011", "ГОСТ 2.307-2011. Нанесение размеров и предельных отклонений",
     "https://meganorm.ru/Data2/1/4293800/4293800223.htm", 25000, "Рисунок 92"),
    ("gost_2_309_73", "ГОСТ 2.309-73. Обозначения шероховатости поверхностей",
     "https://meganorm.ru/Data2/1/4294852/4294852118.htm", 8000, "2.13"),
    ("gost_r_2_316_2023", "ГОСТ Р 2.316-2023. Правила нанесения надписей, технических требований и таблиц на графических документах",
     MEGA + "gost_r_2_316-2023_natsionalnyy_standart_rossiyskoy.html", 12000, "Ключевые слова"),
]


def fetch(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size:
        return
    with requests.Session() as session:
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504]
        )))
        response = session.get(url, timeout=60)
        response.raise_for_status()
    # Страница проверки доступа не должна стать документом или изображением.
    if any(marker in response.content.lower() for marker in
           (b"captcha", b"checking your browser", b"access denied")):
        raise ValueError(f"Вместо документа получена страница проверки доступа: {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(response.content)
    temporary.replace(destination)


def extract(raw: bytes) -> BeautifulSoup:
    soup = BeautifulSoup(raw, "html.parser")
    container = soup.select_one(".WordSection1") or soup.select_one(".contener_doc")
    if container is None:
        raise ValueError("Не найден основной текст стандарта")
    # Удаляем навигацию, справку базы и повторное оглавление сайта.
    for element in container.select("script, style, noscript, .crumbs_span, .open_megainfo, details"):
        element.decompose()
    for element in container.select(".s5"):
        element.decompose()
    for anchor in container.find_all("a"):
        anchor.unwrap()
    # Word HTML содержит переносы строк внутри словосочетаний. Сохраняем
    # границы HTML-абзацев, нормализуя только пробелы внутри текстовых узлов.
    for node in list(container.find_all(string=True)):
        node.replace_with(re.sub(r"\s+", " ", str(node)).replace("\ufeff", ""))
    return container


def process(spec: tuple) -> dict:
    key, title, url, minimum, terminal = spec
    raw_path = SOURCES / "html" / f"{key}.html"
    fetch(url, raw_path)
    raw = raw_path.read_bytes()
    container = extract(raw)
    plain = re.sub(r"\s+", " ", container.get_text(" ", strip=True))
    if len(plain) < minimum or terminal not in plain:
        raise ValueError(f"Неполный текст {key}: {len(plain)} символов; ожидается {terminal}")

    tasks = {}
    illustrations = []
    for index, tag in enumerate(container.find_all("img"), 1):
        image_url = urljoin(url, tag["src"])
        suffix = Path(urlparse(image_url).path).suffix.lower()
        if suffix not in {".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp"}:
            raise ValueError(f"Неожиданный формат иллюстрации: {image_url}")
        filename = sha256(image_url.encode()).hexdigest()[:20] + suffix
        destination = SOURCES / "assets" / key / filename
        tasks[image_url] = destination
        tag["src"] = f"../assets/{key}/{filename}"
        tag["alt"] = tag.get("alt") or f"Изображение {index} из {title.split('. ', 1)[0]}"
        illustrations.append({"url": image_url, "path": destination.relative_to(ROOT).as_posix()})
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda item: fetch(*item), tasks.items()))

    # Статическая локальная копия: без скриптов, рекламы и внешних ресурсов.
    offline = SOURCES / "offline" / f"{key}.html"
    offline.parent.mkdir(parents=True, exist_ok=True)
    offline.write_text(
        '<!doctype html><html lang="ru"><meta charset="utf-8">'
        f"<title>{title}</title><style>body{{max-width:1100px;margin:30px auto;"
        "font:16px/1.5 sans-serif;padding:20px}td{padding:5px;border:1px solid #aaa}"
        "img{max-width:100%;height:auto}.aC{text-align:center}.aR{text-align:right}"
        f"</style><body>{container}</body></html>", encoding="utf-8"
    )
    for tag in container.find_all("img"):
        tag["src"] = "../../sources/gost/" + tag["src"].removeprefix("../")
    body = markdownify(
        str(container), heading_style="ATX", escape_underscores=False,
        keep_inline_images_in=sorted({tag.parent.name for tag in container.find_all("img")}),
    )
    body = re.sub(r"\n{3,}", "\n\n", body).strip()
    # Заголовок содержит название документа для атрибуции; далее идет его текст,
    # а не пересказ. Рисунки не подменяются придуманным OCR или описанием.
    markdown = f"# {title}\n\n{body}\n"
    output = KNOWLEDGE / f"{key}.md"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markdown, encoding="utf-8", newline="\n")
    print(f"{key}: text={len(plain)}, markdown={len(markdown)}, images={len(illustrations)}", flush=True)
    return {
        "standard": title, "source_url": url,
        "raw_html": raw_path.relative_to(ROOT).as_posix(),
        "raw_sha256": sha256(raw).hexdigest(),
        "markdown": output.relative_to(ROOT).as_posix(),
        "markdown_sha256": sha256(markdown.encode()).hexdigest(),
        "text_characters": len(plain), "markdown_characters": len(markdown),
        "tables": len(container.find_all("table")),
        "offline_html": offline.relative_to(ROOT).as_posix(),
        "illustrations": illustrations,
    }


if __name__ == "__main__":
    documents = [process(spec) for spec in DOCUMENTS]
    manifest = {
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "edition_note": "Снимки текстов указанных источников; полнота включения позднейших поправок не подтверждена.",
        "format_note": "Все текстовые разделы и таблицы извлечены из HTML. Графические символы и рисунки сохранены как изображения; RAG индексирует только текст.",
        "documents": documents,
    }
    (SOURCES / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
