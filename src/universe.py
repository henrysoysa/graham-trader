"""Stock-universe options and their DataFetcher index keys.

This is a *pure* module (no Streamlit / no I/O) so it can be imported and
unit-tested on its own. ``app.py`` builds the on-screen picker from these
constants, and every page shares them, so the Screener, Portfolio Simulator
and Backtesting pages always offer — and honour — the same set of markets.

Separator rows (prefixed with ``---``) are display-only and have no mapping.
"""
from typing import List, Optional

# Order matters: this is the exact order shown in the dropdown.
UNIVERSE_OPTIONS: List[str] = [
    "--- Developed Markets (Large Cap) ---",
    "S&P 500 (USA)",
    "NASDAQ-100 (USA)",
    "Dow 30 (USA)",
    "FTSE 100 (UK)",
    "DAX (Germany)",
    "CAC 40 (France)",
    "Nikkei 225 (Japan)",
    "ASX 200 (Australia)",
    "TSX 60 (Canada)",
    "--- Emerging Markets (Large Cap) ---",
    "India - NIFTY 50",
    "India - SENSEX 30",
    "Brazil - BOVESPA",
    "China - Hong Kong (HSI)",
    "China - US ADRs",
    "South Africa - JSE Top 40",
    "Mexico - IPC",
    "Indonesia - IDX",
    "Vietnam - VN30",
    "Emerging Markets ADRs (Recommended)",
    "--- Broader Developed Markets ---",
    "STOXX Europe 600",
    "FTSE 250 (UK Mid Cap)",
    "TOPIX Core 30 (Japan)",
    "KOSPI 200 (South Korea)",
    "Taiwan 50",
    "Straits Times Index (Singapore)",
    "SMI (Switzerland)",
    "--- High-Growth Emerging Economies 🚀 ---",
    "India - NIFTY Next 50",
    "India - Smallcap 100",
    "Vietnam - VN100",
    "Philippines - PSEi",
    "Thailand - SET50",
    "Malaysia - KLCI",
    "Bangladesh - DSE (limited data)",
    "Egypt - EGX 30",
    "Saudi Arabia - TASI",
    "UAE - ADX / DFM",
    "Turkey - BIST 100",
    "Poland - WIG 20",
    "Pakistan - KSE 100 (limited data)",
    "Growth Markets ADRs (Recommended) 🚀",
    "--- Hidden Gems (Small/Mid Cap) 💎 ---",
    "Russell 2000 (US Small Caps)",
    "India - Mid Caps",
    "Brazil - Small Caps",
    "China - Small Cap ADRs",
    "Emerging Markets Small Caps 🌟",
    "--- Other ---",
    "Custom Tickers",
]

# Display label -> DataFetcher.get_index_tickers() key.
UNIVERSE_MAPPING = {
    # Developed Markets
    "S&P 500 (USA)": "SP500",
    "NASDAQ-100 (USA)": "NASDAQ100",
    "Dow 30 (USA)": "DOW30",
    "FTSE 100 (UK)": "FTSE100",
    "DAX (Germany)": "DAX",
    "CAC 40 (France)": "CAC40",
    "Nikkei 225 (Japan)": "NIKKEI225",
    "ASX 200 (Australia)": "ASX200",
    "TSX 60 (Canada)": "TSX60",
    # Emerging Markets
    "India - NIFTY 50": "INDIA_NIFTY50",
    "India - SENSEX 30": "INDIA_SENSEX",
    "Brazil - BOVESPA": "BRAZIL_BOVESPA",
    "China - Hong Kong (HSI)": "CHINA_HSI",
    "China - US ADRs": "CHINA_ADR",
    "South Africa - JSE Top 40": "SOUTHAFRICA_TOP40",
    "Mexico - IPC": "MEXICO_IPC",
    "Indonesia - IDX": "INDONESIA_IDX",
    "Vietnam - VN30": "VIETNAM_VN30",
    "Emerging Markets ADRs (Recommended)": "EMERGING_ADR",
    # Small/Mid Cap Discovery
    "Russell 2000 (US Small Caps)": "RUSSELL2000",
    "India - Mid Caps": "INDIA_MIDCAP",
    "Brazil - Small Caps": "BRAZIL_SMALLCAP",
    "China - Small Cap ADRs": "CHINA_SMALLCAP",
    "Emerging Markets Small Caps 🌟": "EMERGING_SMALLCAP",
    # Broader Developed Markets
    "STOXX Europe 600": "STOXX600",
    "FTSE 250 (UK Mid Cap)": "FTSE250",
    "TOPIX Core 30 (Japan)": "TOPIX_CORE30",
    "KOSPI 200 (South Korea)": "KOSPI200",
    "Taiwan 50": "TAIWAN50",
    "Straits Times Index (Singapore)": "STI_SINGAPORE",
    "SMI (Switzerland)": "SMI_SWISS",
    # High-Growth Emerging Economies
    "India - NIFTY Next 50": "INDIA_NIFTY_NEXT50",
    "India - Smallcap 100": "INDIA_SMALLCAP100",
    "Vietnam - VN100": "VIETNAM_VN100",
    "Philippines - PSEi": "PHILIPPINES_PSEI",
    "Thailand - SET50": "THAILAND_SET50",
    "Malaysia - KLCI": "MALAYSIA_KLCI",
    "Bangladesh - DSE (limited data)": "BANGLADESH_DSE",
    "Egypt - EGX 30": "EGYPT_EGX30",
    "Saudi Arabia - TASI": "SAUDI_TASI",
    "UAE - ADX / DFM": "UAE_ADX_DFM",
    "Turkey - BIST 100": "TURKEY_BIST100",
    "Poland - WIG 20": "POLAND_WIG20",
    "Pakistan - KSE 100 (limited data)": "PAKISTAN_KSE100",
    "Growth Markets ADRs (Recommended) 🚀": "GROWTH_MARKETS_ADR",
}

UNIVERSE_HELP = (
    "🚀 = High-growth economies (IMF 2026–2030 projections). "
    "💎 = Hidden Gems. ADRs = US-listed, easiest to trade."
)


def is_separator(option: str) -> bool:
    """True for display-only separator rows."""
    return option.startswith("---")


def resolve_index_key(option: str) -> Optional[str]:
    """Return the DataFetcher index key for a market label.

    Returns ``None`` for the custom-tickers row, separator rows, or any label
    that has no mapping (which should never happen — the test suite asserts
    every non-separator, non-custom option is mapped).
    """
    if option == "Custom Tickers" or is_separator(option):
        return None
    return UNIVERSE_MAPPING.get(option)
