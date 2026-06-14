import json

from localevidence.acquire import AcquiredPaper
from localevidence.index import PassageIndex
from localevidence.library import catalog
from localevidence.pack import export_pack, harvest_pack


def _build_index(tmp_path):
    idx = PassageIndex(store_dir=tmp_path / "store")
    papers = []
    for i in range(6):
        f = tmp_path / f"p{i}.txt"
        f.write_text(f"Paper {i} discusses clinical topic number {i} in detail. " * 60)
        catalog.upsert(dict(slug=f"p{i}", doi=f"10.7/{i}", pmid="", title=f"Paper {i} on a topic",
                            authors="A B", year="2020", journal="J Test", tier="rct",
                            source="test", text_path=str(f), pdf_path=""))
        papers.append(AcquiredPaper(slug=f"p{i}", doi=f"10.7/{i}", title=f"Paper {i} on a topic",
                                    text_path=str(f), tier="rct"))
    idx.add_papers(papers, verbose=False)
    return idx


def test_export_pack_structure_and_shareable_boundary(tmp_path):
    idx = _build_index(tmp_path)
    out = tmp_path / "pack"
    summ = export_pack(out, index=idx, verbose=False)
    assert summ["papers"] == 6

    # paper list is bibliographic ONLY — no text/pdf paths (the shareable boundary)
    papers = [json.loads(l) for l in (out / "papers.jsonl").read_text().splitlines() if l.strip()]
    assert len(papers) == 6
    keys = set().union(*[set(p) for p in papers])
    assert "text_path" not in keys and "pdf_path" not in keys
    assert keys <= {"slug", "doi", "pmid", "title", "authors", "year", "journal", "tier", "source"}

    # map: clusters cover every paper; links exist; NO verbatim passage text leaks
    m = json.loads((out / "map.json").read_text())
    covered = {s for c in m["clusters"] for s in c["members"]}
    assert covered == {f"p{i}" for i in range(6)}
    assert m["n_papers"] == 6 and m["links"]
    assert "discusses clinical topic" not in json.dumps(m)   # no verbatim text in the map

    # summaries: empty own-words slots to be filled by `pack summarise`
    sums = [json.loads(l) for l in (out / "summaries.jsonl").read_text().splitlines() if l.strip()]
    assert len(sums) == 6 and all(s["provides"] == "" for s in sums)
    assert (out / "README.md").exists()


def test_harvest_pack_reconstructs(tmp_path, monkeypatch):
    import localevidence.library as L
    out = tmp_path / "pack"
    out.mkdir()
    (out / "papers.jsonl").write_text("\n".join(
        json.dumps({"slug": f"p{i}", "doi": f"10.9/{i}", "title": f"P{i}", "tier": "rct"})
        for i in range(3)))

    def fake_pull(doi, **kw):
        f = tmp_path / ("h-" + doi.replace("/", "-") + ".txt")
        f.write_text("harvested clinical full text body " * 40)
        return {"_status": "pulled", "slug": doi.replace("/", "-"), "doi": doi,
                "title": kw.get("title", ""), "text_path": str(f)}

    monkeypatch.setattr(L, "pull", fake_pull)
    idx = PassageIndex(store_dir=tmp_path / "hstore")
    summ = harvest_pack(out, index=idx, verbose=False)
    assert summ["listed"] == 3 and summ["pulled"] == 3
    assert summ["passages_indexed"] > 0
