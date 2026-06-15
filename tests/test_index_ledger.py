import numpy as np

from localevidence import embedding
from localevidence.acquire import AcquiredPaper
from localevidence.index import PassageIndex, _is_low_value_chunk, _downrank_low_value
from localevidence.ledger import Ledger


def test_dosing_tables_are_kept_not_dropped():
    # the dogfooding bug: dense drug-dosing tables (the most clinically valuable
    # content) were dropped as "low-value numeric tables".
    dose_table = ("Cefotaxime 50 mg/kg IV 6-hourly (max 2 g). Ceftriaxone 100 mg/kg IV "
                  "daily (max 4 g). Benzylpenicillin 60 mg/kg IV. Dexamethasone 0.15 mg/kg "
                  "IV every 6 hours for 4 days (max 10 mg). Vancomycin 60 mg/kg/day.")
    assert _is_low_value_chunk(dose_table) is False          # kept (the fix)
    # genuine junk still dropped:
    refs = ("van de Beek D Lancet 2021 doi.org/10.1/x et al accessed https://a "
            "Hasbun R JAMA 2022 doi.org/10.2/y accessed https://b doi.org/10.3/z")
    assert _is_low_value_chunk(refs) is True                  # reference dump
    grid = " ".join(["12.3", "4.5", "6.7", "8.9", "10.1", "2.2", "3.3"] * 4)
    assert _is_low_value_chunk(grid) is True                  # pure numeric grid, no doses


def test_low_value_catches_vancouver_refs_and_contact_headers():
    # Vancouver numbered refs WITHOUT doi.org / "et al" / "accessed" slipped the
    # old filter — the live E. coli failure: a references chunk out-ranked the body.
    vancouver = ("1. Smith J, Brown K. Acute pyelonephritis in children. Lancet. "
                 "1976;1:490-495. 2. Jones A. Urinary tract infection. JAMA. 1982;3:120-130. "
                 "3. Lee B. Pediatric nephrology. N Engl J Med. 1990;5:55-60. "
                 "4. Park C. Renal scarring. Pediatrics. 2001;7:201-209.")
    assert _is_low_value_chunk(vancouver) is True
    # contact / affiliation header boilerplate (an email + an affiliation cue)
    contact = ("Dr. Vanaja Kumar, Head of Department of Bacteriology, National "
               "Institute for Research, Chennai 600031, India. "
               "E-mail: vanajakumar@example.org")
    assert _is_low_value_chunk(contact) is True
    # real clinical prose that merely mentions a year is NOT flagged
    prose = ("In a 2019 prospective cohort of children with febrile UTI, Escherichia "
             "coli accounted for roughly 80 to 90 percent of isolates, and renal "
             "scarring developed in a minority of confirmed pyelonephritis cases.")
    assert _is_low_value_chunk(prose) is False


def test_downrank_low_value_penalises_only_junk():
    refs = ("1. A. Lancet. 1976;1:490-5. 2. B. JAMA. 1982;3:120-30. "
            "3. C. N Engl J Med. 1990;5:55-60.")
    body = "Escherichia coli is responsible for 80 to 90 percent of paediatric pyelonephritis."
    fused = {1: 0.030, 2: 0.030}
    out = _downrank_low_value(fused, {1: refs, 2: body}, factor=0.2)
    assert out[2] == 0.030                # body untouched
    assert out[1] == 0.030 * 0.2          # reference dump penalised
    assert out[2] > out[1]                # body now outranks the junk


def _inject_legacy(idx, *, slug, text, title="X", tier="article"):
    """Insert a passage straight into the store, bypassing the index-time filter —
    simulates junk indexed before the filter was strengthened (the real corpus)."""
    pid = len(idx.meta)
    idx._con.execute(
        "INSERT INTO passages (passage_id,slug,title,doi,pmid,year,journal,tier,chunk_idx,text)"
        " VALUES (?,?,?,?,?,?,?,?,?,?)",
        (pid, slug, title, "", "", "", "", tier, 99, text))
    idx._con.execute("INSERT INTO fts(rowid,text) VALUES (?,?)", (pid, text))
    idx._con.commit()
    idx.meta.append({"passage_id": pid, "slug": slug, "title": title, "doi": "",
                     "pmid": "", "year": "", "journal": "", "tier": tier, "chunk_idx": 99})
    idx.vectors = np.vstack([idx.vectors, embedding.embed([text])])


def test_search_downranks_legacy_reference_chunk(tmp_path):
    body = tmp_path / "body.txt"
    body.write_text("Escherichia coli causes most paediatric urinary tract infection "
                    "and pyelonephritis in children. " * 40)
    idx = PassageIndex(store_dir=tmp_path / "s")
    idx.add_papers([AcquiredPaper(slug="uti", title="UTI overview",
                                  text_path=str(body), tier="review")], verbose=False)
    # a legacy references chunk for the SAME topic, sharing the query's tokens
    ref = ("1. urinary tract infection pyelonephritis coli children. Lancet. 1976;1:490-5. "
           "2. urinary coli children. JAMA. 1982;3:120-30. "
           "3. pyelonephritis children. N Engl J Med. 1990;5:55-60.")
    _inject_legacy(idx, slug="uti", text=ref)
    res = idx.search("Escherichia coli urinary tract infection pyelonephritis children", k=5)
    assert res
    assert "1976;1:490" not in res[0].text           # the reference dump is NOT on top
    assert any("1976;1:490" in r.text for r in res)   # but still retrievable (down-ranked, not dropped)


def test_index_add_search_and_alignment(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("Diazepam is first-line for alcohol withdrawal in most patients. " * 60)
    b = tmp_path / "b.txt"
    b.write_text("Croup is treated with a single dose of dexamethasone. " * 60)
    idx = PassageIndex(store_dir=tmp_path / "store")
    n = idx.add_papers([
        AcquiredPaper(slug="a", title="A", text_path=str(a), tier="rct"),
        AcquiredPaper(slug="b", title="B", text_path=str(b), tier="review"),
    ], verbose=False)
    assert n > 0
    # dense matrix stays row-aligned with the catalog
    assert idx.vectors.shape[0] == len(idx.meta)
    # the lexical (BM25) arm finds paper a by its drug terms even though the
    # fake embedder gives random dense vectors
    res = idx.search("alcohol withdrawal diazepam", k=5)
    assert res and any(r.slug == "a" for r in res)


def test_index_incremental_skips_indexed_slug(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("word " * 300)
    idx = PassageIndex(store_dir=tmp_path / "s2")
    paper = AcquiredPaper(slug="a", title="A", text_path=str(f), tier="article")
    assert idx.add_papers([paper], verbose=False) > 0
    assert idx.add_papers([paper], verbose=False) == 0   # same slug -> no re-chunk


def test_ledger_record_update_roundtrip(tmp_path):
    led = Ledger(store_dir=tmp_path / "led")
    eid = led.record("What HR threshold admits anorexia nervosa?",
                     evidence=[{"slug": "x", "doi": "10.x/y"}])
    assert led.stats()["questions"] == 1
    assert led.update(eid, answer="Admit if HR<40.", confidence="high")
    e = led.get(eid)
    assert e["answer"] == "Admit if HR<40." and e["confidence"] == "high"
    assert led.stats()["answered"] == 1


def test_ledger_find_similar_excludes_identical(tmp_path):
    led = Ledger(store_dir=tmp_path / "led2")
    led.record("identical question text")
    # identical string is deliberately excluded (it surfaces *different* priors)
    assert led.find_similar("identical question text", threshold=0.99) == []


def test_bare_search_does_not_inject_offtopic_guidelines(tmp_path):
    # An off-topic guideline indexed FIRST, then a long on-topic guideline whose
    # many chunks fill the organic top-k. The off-topic one shares none of the
    # query's lexical tokens, so it can only reach the results via the
    # tier-guarantee — which must NOT fire on a bare (no-focus_slugs) search.
    g = tmp_path / "g.txt"
    g.write_text("anorexia nervosa refeeding medical instability bradycardia management. " * 60)
    a = tmp_path / "a.txt"
    a.write_text("croup dexamethasone severity adrenaline nebulised airway steroid "
                 "children management treatment stridor barking cough. " * 400)
    idx = PassageIndex(store_dir=tmp_path / "store")
    idx.add_papers([
        AcquiredPaper(slug="anorexia-gl", title="Anorexia guideline",
                      text_path=str(g), tier="guideline"),
        AcquiredPaper(slug="croup-gl", title="Croup guideline",
                      text_path=str(a), tier="guideline"),
    ], verbose=False)
    res = idx.search("croup dexamethasone severity adrenaline nebulised", k=6)
    assert any(r.slug == "croup-gl" for r in res)        # on-topic surfaces
    assert all(r.slug != "anorexia-gl" for r in res)     # off-topic NOT injected
