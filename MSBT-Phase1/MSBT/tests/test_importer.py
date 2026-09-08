from pathlib import Path

import pytest

from msbt.importers.csv_importer import import_trade_file, percent_points_to_decimal
from msbt.importers.filename import parse_trade_filename, try_parse_trade_filename
from msbt.models.trade import TradeStatus
from msbt.validation.validate import validate_trades

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "TripleStrategy_AMEX_SAMPLE_2026-09-05.csv"


def test_percent_points_conversion():
    assert abs(percent_points_to_decimal(11.95) - 0.1195) < 1e-12


def test_parse_filename_with_extension():
    p = parse_trade_filename("TripleStrategy_AMEX_SAMPLE_2026-09-05.csv")
    assert p.strategy == "TripleStrategy"
    assert p.exchange == "AMEX"
    assert p.symbol == "SAMPLE"
    assert p.file_date == "2026-09-05"


def test_parse_filename_without_extension():
    p = parse_trade_filename("TripleStrategy_NYSE_ORCL_2026-09-07")
    assert p.strategy == "TripleStrategy"
    assert p.exchange == "NYSE"
    assert p.symbol == "ORCL"
    assert p.file_date == "2026-09-07"


def test_parse_filename_bad():
    with pytest.raises(ValueError):
        parse_trade_filename("badname.csv")
    assert try_parse_trade_filename("random-export.csv") is None


def test_import_sample_fixture():
    trades, issues = import_trade_file(FIXTURE)
    trades, report = validate_trades(trades, prior_issues=issues, files_uploaded=1)
    assert report.valid_trades >= 24
    completed = [t for t in trades if t.status == TradeStatus.COMPLETED]
    opens = [t for t in trades if t.status == TradeStatus.OPEN]
    assert len(opens) == 1
    assert opens[0].sell_date is None
    t1 = next(t for t in completed if t.source_trade_number == 1)
    assert abs(t1.return_pct - 0.1195) < 1e-12
    assert t1.buy_date.isoformat() == "2008-02-08"
    assert t1.sell_date.isoformat() == "2008-04-25"


def test_import_orcl_style_name_bytes():
    data = FIXTURE.read_bytes()
    trades, issues = import_trade_file(data, filename="TripleStrategy_NYSE_ORCL_2026-09-07.csv")
    assert not any(i["severity"] == "error" and "does not match" in i["message"] for i in issues)
    assert all(t.symbol == "ORCL" for t in trades)
