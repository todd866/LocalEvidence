import pytest

from localevidence.library import catalog
from localevidence.library import providers as P
from localevidence.library.providers import (
    default_providers, load_shadow_provider, pull,
    ShadowProvider, LocalFileProvider,
)

PDF = b"%PDF-1.4 minimal body"

# An operator may drop a git-ignored providers/private.py (the full-stack plug-in).
# When present, the "defaults to stub" behaviour is intentionally overridden, so
# the stub-default tests skip. In a clean checkout / CI there is no private.py.
from pathlib import Path as _Path
_HAS_PRIVATE = (_Path(P.__file__).parent / "private.py").exists()
_skip_if_plugin = pytest.mark.skipif(_HAS_PRIVATE, reason="operator private.py plug-in present")


class FakeProvider:
    """A test provider returning a preset value / raising a preset error."""
    def __init__(self, source="fake", out=None, exc=None, collision_prone=False):
        self.source, self.legal, self.collision_prone = source, False, collision_prone
        self._out, self._exc = out, exc

    def fetch(self, doi, **_kw):
        if self._exc:
            raise self._exc
        return self._out


def test_default_cascade_is_legal_then_shadow():
    assert [p.source for p in default_providers()] == \
        ["localfile", "unpaywall", "europepmc", "shadow"]
    # oa_only drops the shadow tier entirely
    assert [p.source for p in default_providers(oa_only=True)] == \
        ["localfile", "unpaywall", "europepmc"]


def test_shadow_stub_raises_not_implemented():
    with pytest.raises(NotImplementedError):
        ShadowProvider().fetch("10.1/x", title="t")


@_skip_if_plugin
def test_load_shadow_defaults_to_stub(monkeypatch):
    monkeypatch.delenv("LOCALEVIDENCE_SHADOW", raising=False)
    assert isinstance(load_shadow_provider(), ShadowProvider)


@_skip_if_plugin
def test_load_shadow_bogus_env_falls_back_to_stub(monkeypatch):
    monkeypatch.setenv("LOCALEVIDENCE_SHADOW", "no_such_pkg_zzz:Bogus")
    assert isinstance(load_shadow_provider(), ShadowProvider)


def test_unpaywall_placeholder_email_warns_and_skips(capsys, monkeypatch):
    import localevidence.library.providers.unpaywall as up
    up._warned = False
    out = up.UnpaywallProvider(email="you@example.com").fetch("10.1/x", title="t")
    assert out is None
    assert "real contact email" in capsys.readouterr().err


def test_pull_already_have_short_circuits():
    catalog.upsert(dict(slug="10-have-x", doi="10.have/x", text_path="",
                        pdf_path="", title="H", source="test"))
    assert pull("10.have/x")["_status"] == "already_have"


def test_pull_stores_first_title_match(monkeypatch):
    monkeypatch.setattr(P, "pdf_matches_title", lambda b, t: (True, 1.0))
    monkeypatch.setattr(P.catalog, "store_pdf",
                        lambda pdf, **k: {"slug": "s", "source": k.get("source")})
    res = pull("10.new/win", title="t", providers=[FakeProvider(out=PDF)])
    assert res["_status"] == "pulled" and res["source"] == "fake"


def test_pull_rejects_wrong_paper(monkeypatch):
    # title never matches -> wrong_paper_only, never stored
    monkeypatch.setattr(P, "pdf_matches_title", lambda b, t: (False, 0.2))
    res = pull("10.new/wrong", title="some real title",
               providers=[FakeProvider(out=PDF)])
    assert res["_status"] == "wrong_paper_only"
    assert res["best_ratio"] == 0.2


def test_pull_surfaces_provider_errors():
    res = pull("10.new/err", title="t",
               providers=[FakeProvider(exc=RuntimeError("dns boom"))])
    assert res["_status"] == "not_found"
    assert res.get("errors") and "dns boom" in res["errors"][0]["error"]


def test_pull_tries_multiple_candidates(monkeypatch):
    state = {"n": 0}

    def match(_b, _t):
        state["n"] += 1
        return (state["n"] >= 2, 1.0 if state["n"] >= 2 else 0.1)

    monkeypatch.setattr(P, "pdf_matches_title", match)
    monkeypatch.setattr(P.catalog, "store_pdf",
                        lambda pdf, **k: {"slug": "s", "source": k.get("source")})
    # provider yields two candidates; first fails the title check, second wins
    res = pull("10.multi/x", title="t", providers=[FakeProvider(out=[PDF, PDF])])
    assert res["_status"] == "pulled" and state["n"] == 2


def test_localfile_provider_reads_inbox():
    (catalog.INBOX / "10-inbox-x.pdf").write_bytes(PDF)
    got = LocalFileProvider().fetch("10.inbox/x", title="t")
    assert got == PDF
