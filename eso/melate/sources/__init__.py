from .loteria_nacional import fetch_official_history, load_official_file, parse_official_csv_bytes
from .melate_e import MelateEClient, parse_prize_page

__all__ = ["fetch_official_history", "load_official_file", "parse_official_csv_bytes", "MelateEClient", "parse_prize_page"]
