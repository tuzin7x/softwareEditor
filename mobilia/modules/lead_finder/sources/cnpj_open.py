"""Base aberta de CNPJ da Receita Federal (dados abertos, uso livre) importada para um índice local.

A Receita publica mensalmente os arquivos ``Estabelecimentos*.zip``, ``Empresas*.zip`` e ``Municipios.zip``
(CSV com ``;``, codificação latin-1, sem cabeçalho). O usuário baixa os arquivos na página oficial e escolhe a
pasta; o importador lê em fluxo (sem descompactar no disco), guarda **só** os estabelecimentos **ativos** com
CNAE do setor imobiliário e das UFs escolhidas, e monta ``cnpj_index.sqlite3``. A pesquisa consulta esse índice
sem internet.

Minimização (LGPD): o e-mail e os dados de sócios não são importados. Empresários individuais/MEI (natureza
jurídica 2135, 4014, 4081) ficam fora da pesquisa por padrão: o nome deles costuma ser o nome de uma pessoa.
"""
from __future__ import annotations

import csv
import io
import sqlite3
import zipfile
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .. import keywords as kw
from ..geo import UF_NAME, fold, scope_ufs
from ..models import Lead, SearchQuery, SourcePolicy, now_iso
from .base import Estimate, Source, SourceUnavailable, Unit

OFFICIAL_PAGE = "https://dados.gov.br/dados/conjuntos-dados/cadastro-nacional-da-pessoa-juridica---cnpj"
INDIVIDUAL_NATUREZAS = ("2135", "4014", "4081")
ACTIVE = "02"
INDEX_NAME = "cnpj_index.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS estab(
  cnpj TEXT PRIMARY KEY, basico TEXT, fantasia TEXT, razao TEXT DEFAULT '', natureza TEXT DEFAULT '',
  nome_fold TEXT, uf TEXT, municipio TEXT, municipio_fold TEXT, endereco TEXT, cep TEXT,
  tel1 TEXT, tel2 TEXT, cnae TEXT, cnaes TEXT, inicio TEXT);
CREATE INDEX IF NOT EXISTS ix_estab_loc ON estab(uf, municipio_fold);
CREATE INDEX IF NOT EXISTS ix_estab_basico ON estab(basico);
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
"""


@dataclass
class ImportReport:
    rows_read: int = 0
    kept: int = 0
    with_legal_name: int = 0
    ufs: tuple[str, ...] = ()
    files: tuple[str, ...] = ()


def find_files(folder: Path) -> dict[str, list[Path]]:
    """Localiza os arquivos da Receita numa pasta (zip ou já extraídos)."""
    out: dict[str, list[Path]] = {"estab": [], "empresas": [], "municipios": []}
    for p in sorted(Path(folder).iterdir()):
        n = p.name.lower()
        if not p.is_file():
            continue
        if n.startswith("estabelecimento") or "estabele" in n:
            out["estab"].append(p)
        elif n.startswith("empresa") or "emprecsv" in n:
            out["empresas"].append(p)
        elif n.startswith("municipio") or "municcsv" in n:
            out["municipios"].append(p)
    return out


def _streams(path: Path) -> Iterator[tuple[io.BufferedIOBase, int]]:
    """Fluxos binários (e tamanho) de um arquivo da Receita: cada membro do zip ou o próprio arquivo."""
    if zipfile.is_zipfile(path):
        with zipfile.ZipFile(path) as zf:
            for info in zf.infolist():
                if not info.is_dir():
                    with zf.open(info) as raw:
                        yield raw, max(1, info.file_size)
    else:
        with open(path, "rb") as raw:
            yield raw, max(1, path.stat().st_size)


def _rows(raw) -> Iterator[list[str]]:
    text = io.TextIOWrapper(raw, encoding="latin-1", newline="")
    yield from csv.reader(text, delimiter=";", quotechar='"')


def _tel(ddd: str, num: str) -> str:
    ddd, num = (ddd or "").strip(), (num or "").strip()
    return f"{ddd}{num}" if ddd and num else ""


def open_index(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path))
    con.execute("PRAGMA journal_mode=WAL")
    con.executescript(_SCHEMA)
    return con


def import_open_data(folder: Path, index_path: Path, ufs: list[str] | None, ctx, progress=None) -> ImportReport:
    """Importa os arquivos da Receita para o índice local. ``ufs`` vazio = Brasil inteiro.

    Os dados das UFs importadas são trocados de uma vez no fim; cancelar não deixa o índice pela metade."""
    files = find_files(folder)
    if not files["estab"]:
        raise SourceUnavailable("Nenhum arquivo Estabelecimentos*.zip encontrado na pasta escolhida.")
    if not files["municipios"]:
        raise SourceUnavailable("Falta o arquivo Municipios.zip (nomes das cidades) na pasta escolhida.")
    want_ufs = {u for u in (ufs or []) if u in UF_NAME}
    cnaes = set(kw.CNAE)
    report = ImportReport(ufs=tuple(sorted(want_ufs)) or ("BR",),
                          files=tuple(p.name for v in files.values() for p in v))
    say = progress or (lambda f, m: None)

    munis: dict[str, str] = {}
    for p in files["municipios"]:
        for raw, _ in _streams(p):
            for r in _rows(raw):
                if len(r) >= 2:
                    munis[r[0].strip()] = r[1].strip().title()

    con = open_index(index_path)
    try:
        con.execute("DROP TABLE IF EXISTS estab_new")
        con.execute("CREATE TABLE estab_new AS SELECT * FROM estab WHERE 0")
        batch: list[tuple] = []
        n_files = len(files["estab"])
        for i, p in enumerate(files["estab"]):
            for raw, size in _streams(p):
                for r in _rows(raw):
                    report.rows_read += 1
                    if report.rows_read % 20000 == 0:
                        ctx.check()
                        try:
                            frac = min(1.0, raw.tell() / size)
                        except (OSError, ValueError, AttributeError):
                            frac = 0.0
                        say(0.85 * (i + frac) / n_files,
                            f"Lendo {p.name}: {report.rows_read:,} linhas, {report.kept:,} do setor".replace(",", "."))
                    if len(r) < 25 or r[5].strip() != ACTIVE:
                        continue
                    uf = r[19].strip().upper()
                    if want_ufs and uf not in want_ufs:
                        continue
                    principal = r[11].strip()
                    secondary = [c.strip() for c in r[12].split(",") if c.strip()]
                    matched = [c for c in [principal, *secondary] if c in cnaes]
                    if not matched:
                        continue
                    cnpj = f"{r[0].strip()}{r[1].strip()}{r[2].strip()}"
                    street = " ".join(x for x in (r[13].strip(), r[14].strip(), r[15].strip()) if x)
                    endereco = ", ".join(x for x in (street, r[16].strip(), r[17].strip()) if x).title()
                    muni = munis.get(r[20].strip(), r[20].strip())
                    fantasia = r[4].strip()
                    batch.append((cnpj, r[0].strip(), fantasia, "", "", fold(fantasia), uf, muni, fold(muni),
                                  endereco, r[18].strip(), _tel(r[21], r[22]), _tel(r[23], r[24]),
                                  principal if principal in cnaes else matched[0], " " + " ".join(matched) + " ",
                                  r[10].strip()))
                    if len(batch) >= 5000:
                        con.executemany("INSERT OR REPLACE INTO estab_new VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                                        batch)
                        report.kept += len(batch); batch.clear()
        if batch:
            con.executemany("INSERT OR REPLACE INTO estab_new VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", batch)
            report.kept += len(batch); batch.clear()
        con.execute("CREATE INDEX IF NOT EXISTS ix_new_basico ON estab_new(basico)")
        con.commit()

        if files["empresas"]:
            basicos = {b for (b,) in con.execute("SELECT DISTINCT basico FROM estab_new")}
            updates: list[tuple] = []
            n_emp = len(files["empresas"])
            for i, p in enumerate(files["empresas"]):
                say(0.85 + 0.13 * i / n_emp, f"Lendo razões sociais ({p.name})")
                for raw, _ in _streams(p):
                    for k, r in enumerate(_rows(raw)):
                        if k % 50000 == 0:
                            ctx.check()
                        if len(r) >= 3 and r[0].strip() in basicos:
                            razao = r[1].strip()
                            updates.append((razao, r[2].strip(), fold(razao), r[0].strip()))
                            if len(updates) >= 5000:
                                con.executemany("UPDATE estab_new SET razao=?, natureza=?, "
                                                "nome_fold=trim(coalesce(nome_fold, '') || ' ' || ?) WHERE basico=?",
                                                updates)
                                updates.clear()
            if updates:
                # mesma dobra (sem acentos, minúsculas) usada nos termos de busca: o lower() do SQLite só trata ASCII
                con.executemany("UPDATE estab_new SET razao=?, natureza=?, "
                                "nome_fold=trim(coalesce(nome_fold, '') || ' ' || ?) WHERE basico=?", updates)
            con.commit()
        ctx.check()
        say(0.99, "Gravando o índice")
        with con:
            if want_ufs:
                con.execute(f"DELETE FROM estab WHERE uf IN ({','.join('?' * len(want_ufs))})", sorted(want_ufs))
            else:
                con.execute("DELETE FROM estab")
            con.execute("INSERT OR REPLACE INTO estab SELECT * FROM estab_new")
            con.execute("DROP TABLE estab_new")
            prev = dict(con.execute("SELECT key, value FROM meta"))
            all_ufs = sorted(set(prev.get("ufs", "").split(",")) - {""} | want_ufs) if want_ufs else ["BR"]
            con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)",
                            [("imported_at", now_iso()), ("ufs", ",".join(all_ufs)),
                             ("files", ", ".join(report.files))])
        report.with_legal_name = con.execute("SELECT count(*) FROM estab WHERE razao != ''").fetchone()[0]
        return report
    except BaseException:
        con.rollback()
        con.execute("DROP TABLE IF EXISTS estab_new")
        con.commit()
        raise
    finally:
        con.close()


def index_info(index_path: Path) -> dict:
    """Resumo do índice local (vazio se não existir)."""
    if not Path(index_path).is_file():
        return {}
    con = sqlite3.connect(str(index_path))
    try:
        meta = dict(con.execute("SELECT key, value FROM meta"))
        meta["rows"] = con.execute("SELECT count(*) FROM estab").fetchone()[0]
        meta["sem_natureza"] = con.execute("SELECT count(*) FROM estab WHERE natureza = ''").fetchone()[0]
        return meta
    except sqlite3.Error:
        return {}
    finally:
        con.close()


class CnpjOpenSource(Source):
    id = "cnpj_aberto"
    title = "Base aberta de CNPJ"
    online = False
    description = "Cadastro oficial da Receita Federal (dados abertos), importado para o seu computador."
    policy = SourcePolicy(store_days=None, exportable=True,
                          attribution="Receita Federal do Brasil — Cadastro Nacional da Pessoa Jurídica (dados abertos)",
                          terms_url=OFFICIAL_PAGE)

    @property
    def index_path(self) -> Path:
        return Path(self.data_dir) / INDEX_NAME

    def availability(self):
        if self.settings is not None and not self.settings.cnpj_enabled:
            return False, "Desativada nas configurações."
        info = index_info(self.index_path)
        if not info.get("rows"):
            return False, "Importe os arquivos da Receita em Configurações › Base aberta de CNPJ."
        ufs = info.get("ufs", "")
        text = f"{info['rows']:,} empresas no índice local ({'Brasil' if ufs == 'BR' else ufs})"
        if info.get("sem_natureza"):
            text += (f" · {info['sem_natureza']:,} sem natureza jurídica (falta Empresas*.zip): ficam fora da busca, "
                     "a menos que inclua empresários individuais")
        return True, text.replace(",", ".")

    def plan(self, query: SearchQuery) -> list[Unit]:
        cnaes = kw.cnaes_for(query.keywords)
        terms = kw.custom_terms(query.keywords)
        if not cnaes and not terms:
            return []
        return [Unit("índice local de CNPJ", {"cnaes": cnaes, "terms": terms}, 0)]

    def estimate(self, query: SearchQuery) -> Estimate:
        return Estimate(len(self.plan(query)), 0, 0.0, "Consulta local, sem custo e sem internet")

    def fetch(self, unit: Unit, query: SearchQuery, ctx):
        where, args = [], []
        scope = query.scope
        if scope.kind == "municipios":
            pairs = [(u, fold(m)) for m, u in scope.municipalities]
            where.append("(" + " OR ".join("(uf=? AND municipio_fold=?)" for _ in pairs) + ")")
            for p in pairs:
                args.extend(p)
        elif scope.kind != "pais":
            ufs = scope_ufs(scope)
            where.append(f"uf IN ({','.join('?' * len(ufs))})")
            args.extend(ufs)
        conds = []
        for c in unit.params["cnaes"]:
            conds.append("cnaes LIKE ?"); args.append(f"% {c} %")
        for t in unit.params["terms"]:
            conds.append("nome_fold LIKE ?"); args.append(f"%{t}%")
        where.append("(" + " OR ".join(conds) + ")")
        if not query.include_individuals:
            # natureza desconhecida (Empresas*.zip ausente ou incompleto) também fica de fora: pode ser um MEI
            where.append(f"natureza != '' AND natureza NOT IN ({','.join('?' * len(INDIVIDUAL_NATUREZAS))})")
            args.extend(INDIVIDUAL_NATUREZAS)
        if not self.index_path.is_file():
            raise SourceUnavailable("O índice local de CNPJ ainda não foi importado.")
        con = sqlite3.connect(f"file:{self.index_path}?mode=ro", uri=True)
        try:
            cur = con.execute("SELECT cnpj, fantasia, razao, natureza, uf, municipio, endereco, cep, tel1, tel2, cnae "
                              f"FROM estab WHERE {' AND '.join(where)} ORDER BY uf, municipio_fold", args)
            for k, (cnpj, fantasia, razao, natureza, uf, muni, end, cep, t1, t2, cnae) in enumerate(cur):
                if k % 2000 == 0:
                    ctx.check()
                name = fantasia or razao or f"CNPJ {cnpj}"
                yield Lead(source=self.id, source_ref=cnpj, name=name.title() if name.isupper() else name,
                           legal_name=razao, category=kw.classify(name, cnae=cnae), address=end, city=muni, uf=uf,
                           cep=cep, phone=t1, phone2=t2, cnpj=cnpj, cnae=cnae, situacao="ATIVA",
                           natureza=natureza, query=unit.label)
        finally:
            con.close()
