import os
import re
import unicodedata
import warnings
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest, proportion_confint
from statsmodels.stats.contingency_tables import Table2x2
import statsmodels.api as sm
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")


INPUT_XLSX = "/mnt/user-data/outputs/UFSC_aprovados_por_ano.xlsx"
OUTDIR = "/mnt/user-data/outputs"

SOBRENOMES = ["SILVA", "SANTOS", "OLIVEIRA", "SOUZA", "PEREIRA"]
ANOS = ["2003", "2004", "2005", "2023", "2024", "2025"]
GRUPOS = {"2003": "Antigo", "2004": "Antigo", "2005": "Antigo",
          "2023": "Recente", "2024": "Recente", "2025": "Recente"}
CURSOS = ["ARQUITETURA E URBANISMO", "DIREITO", "MEDICINA"]


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s)
                   if unicodedata.category(c) != "Mn")


def norm(s: str) -> str:
    return strip_accents(s).upper().strip()


_FEMALE = set("""
ABIGAIL ADELICE ADRIANE ALANIS ALICE ALINE AMABILE ANDRIELLE ANELISE ANGELES
ANI ANIELLE ANNELIESE ANNELYN ASHLEY BEATRIZ CAMILLE CANDICE CARECIANE CARLISE
CAROL CAROLINE CAROLYNE CATHERINE CHARLINE CIBELE CLARICE CRISTIANE CRISTINE
DANIELE DAYANE DENISE DIENIFFER DJENNIFER ELAINE ELIZABETH EMANUELI EMANUELLE
EMILIE EMILLY EMILY ENAYLLE EVELYN FABIANE FLAVIANE FRANCIELI FRANCIELLE
FRANCIELLY GABRIELE GABRIELI GABRIELLE GABRIELY GIANNE GISELE GISLENE GRACE
GRAZIELE GREYCE HARIANY HELEN HELLEN HELOISE IASMINE INGRID ISABEL ISABELE
ISABELLE ISABELLI ISIS ISSIS IZABELLE JACQUELINE JAKCELY JAMILE JANICE JAQUELINE
JAYNE JOICE JOSIANI JOYCE JULIANY JULLY KAILANY KAREN KARINE KAROLINE KATHERINE
KATHYNE KATIANE KATLIN KELI KELLY KELY KEMYLLY KETLEY KETLIN LAISE LECIANE LIRIS
LIZ LOUISE LUANI MABEL MAIARI MARIAH MARIANE MARIANNE MARIELE MICHELLE MICHELLY
MILEINE MILENE MIRELLE MONIQUE NADIEJE NAIANE NATALIE NAYARE NICOLE NICOLLY NOELI
NURIELI QUEREN RACHEL RAQUEL REGIANE SABINE SARAH SIMONE SIMONY STEFANE STEFANIE
STEFANY STHEFANY TAISI TALLULAH TATIANE THAIS THAISE THAMIRES THAYS THYCIANE
VIVIAN VIVIANE YASMIM YASMIN
""".split())

_MALE = set("""
ABNER ADAILTON ADAIRTON ADELAR ADEMAR AGUSTIN ALAN ALEX ALEXANDRE ALLAN ALZEMIR
ANDERSON ANDRE ANDREI ANDREW ANDREY ANTHONY ARTHUR ARTUR AUDAIR AXYL BENJAMIN
BRENER BYRON CARLOS CAUAN CAUE CESAR CID CLAUS CRISTOPHER DANIEL DANTE DARTAGNAN
DAVI DENIS DEYVID DIEFERSON DOUGLAS EDGAR EDILSON ELIAS EMANUEL ENDERSON ENDRYW
ERIC ERICKSEN EVERTON FELIPE FELIPPE FELLIPE FILIPE GABRIEL GABRYEL GEORGE
GIOVANI GIOVANNI GLAUBER GREGORY GUILHERME GUNTHER HEITOR HENRI HENRIQUE HERCULES
HILBERT HUILQUES IAN IGOR ISAQUE ISMAEL ISRAEL IURI IURY JAKE JARED JAVIER JAYME
JEAN JHONATAN JOAQUIM JONAS JONATAN JONATHAN JORDAN JORGE JOSE JOSUE JUAN KALIL
KAUAN KAUET KAUI KELVYN KLAUS KLEBER KURT LEANDER LEONEL LUAN LUCAS LUIS LUIZ
MAICON MAIKE MANOEL MARCEL MARCELL MARCOS MARCUS MARLON MARLUS MARTIN MATEUS
MATHEUS MATHIAS MATIAS MATTHIEU MAX MEGARON MEIERSON MICHAEL MICHEL MIGUEL MILTON
MOISES NATHANAEL NEAL NEEMIAS NELSON NICOLAS OLIVER OSCAR PATRICK RAFAEL RALFF
RALPH RAUL RENAN RENE ROBSON RONALD RONNY RUAN RUBENS RUI RUY SAMUEL TEILOR
THALES TIARAJU TUAN VANDERLEI VICTOR VINICIUS VITHOR VITOR VOLMIR VOLNEI WAGNER
WALDIR WALLACE WALTER WERNER WESLEY WILIAM WILLIAM WILLIAN
CAUA LUCCA KOTA
""".split())  

_UNKNOWN = set("""
ANGEL ARIEL CARMINE CLAUDE ELLIS JACY MARIEL NIRENI NOLCI REVIS SHAMYL SIRAN
CAUANE
""".split())


def infer_sexo(nome: str) -> str:
    """Retorna 'M', 'F' ou 'U' (desconhecido) a partir do primeiro nome."""
    first = norm(nome).split()[0]
    if first in _UNKNOWN:
        return "U"
    if first in _MALE:
        return "M"
    if first in _FEMALE:
        return "F"
    if first.endswith("A"):
        return "F"
    if first.endswith("O"):
        return "M"
    return "U"



def load_data(path: str = INPUT_XLSX) -> pd.DataFrame:
    rows = []
    sheets = pd.read_excel(path, sheet_name=None, header=None)
    for ano, df in sheets.items():
        ano = str(ano).strip()
        if ano not in GRUPOS:
            continue
        curso_atual = None
        for _, r in df.iterrows():
            a = r[0]
            b = r[1] if len(r) > 1 else None
            if not isinstance(a, str):
                continue
            txt = a.strip()
            U = norm(txt)
            # cabecalho de curso?
            if "ARQUITETURA" in U:
                curso_atual = "ARQUITETURA E URBANISMO"; continue
            if U.startswith("DIREITO"):
                curso_atual = "DIREITO"; continue
            if U.startswith("MEDICINA"):
                curso_atual = "MEDICINA"; continue
            if U.startswith("UFSC") or U.startswith("NOME") or U == "":
                continue
            
            if curso_atual and isinstance(b, (str, int, float)) and str(b).strip():
                nome = txt
                insc = str(b).strip()
                if insc.lower() == "inscricao" or insc.lower() == "inscrição":
                    continue
                rows.append((ano, GRUPOS[ano], curso_atual, nome, insc))

    df = pd.DataFrame(rows, columns=["ano", "grupo", "curso", "nome", "inscricao"])
    df["sexo"] = df["nome"].map(infer_sexo)
    df["nome_norm"] = df["nome"].map(norm)
    for s in SOBRENOMES:
        pat = re.compile(r"(?<![A-Z])" + s + r"(?![A-Z])")
        df[s] = df["nome_norm"].map(lambda x: bool(pat.search(x)))
    df["QUALQUER5"] = df[SOBRENOMES].any(axis=1)
    return df


#realização das estatísticas
def comparar_proporcoes(succ_old, n_old, succ_new, n_new):
    """Compara duas proporcoes; retorna dict completo de metricas e testes."""
    p_old = succ_old / n_old if n_old else np.nan
    p_new = succ_new / n_new if n_new else np.nan
    out = {
        "n_antigo": n_old, "casos_antigo": succ_old, "prop_antigo": p_old,
        "n_recente": n_new, "casos_recente": succ_new, "prop_recente": p_new,
        "dif_pp": (p_new - p_old) * 100,
        "variacao_rel_%": ((p_new - p_old) / p_old * 100) if p_old else np.nan,
    }
    # IC 95% de cada proporcao (Wilson)
    out["ic_antigo"] = proportion_confint(succ_old, n_old, method="wilson") if n_old else (np.nan, np.nan)
    out["ic_recente"] = proportion_confint(succ_new, n_new, method="wilson") if n_new else (np.nan, np.nan)
    # teste z de duas proporcoes
    try:
        z, pz = proportions_ztest([succ_new, succ_old], [n_new, n_old])
    except Exception:
        z, pz = np.nan, np.nan
    out["z"] = z
    out["p_ztest"] = pz

    a, b = succ_old, n_old - succ_old
    c, d = succ_new, n_new - succ_new
    table = np.array([[a, b], [c, d]])
    out["tabela"] = table

    chi2, pchi, dof, exp = stats.chi2_contingency(table, correction=True)
    out["chi2"] = chi2
    out["p_chi2"] = pchi
    out["esperada_min"] = exp.min()
   
    if exp.min() < 5:
        _, pf = stats.fisher_exact(table)
        out["p_fisher"] = pf
        out["p_principal"] = pf
        out["teste_principal"] = "Fisher exato"
    else:
        out["p_fisher"] = np.nan
        out["p_principal"] = pchi
        out["teste_principal"] = "Qui-quadrado (Yates)"
  
    try:
        t2 = Table2x2(np.array([[c, d], [a, b]]))  
        out["RR"] = t2.riskratio
        out["RR_ic"] = tuple(t2.riskratio_confint())
        out["OR"] = t2.oddsratio
        out["OR_ic"] = tuple(t2.oddsratio_confint())
    except Exception:
        out["RR"] = out["OR"] = np.nan
        out["RR_ic"] = out["OR_ic"] = (np.nan, np.nan)
    out["sig"] = "SIM" if (out["p_principal"] is not None and out["p_principal"] < 0.05) else "nao"
    out["direcao"] = "AUMENTOU" if out["dif_pp"] > 0 else ("DIMINUIU" if out["dif_pp"] < 0 else "igual")
    return out


def cochran_armitage(counts, totals, scores):
    """Teste/Cochran-Armitage."""
    counts = np.asarray(counts, float)
    totals = np.asarray(totals, float)
    scores = np.asarray(scores, float)
    N = totals.sum()
    R = counts.sum()
    if N == 0 or R == 0 or R == N:
        return np.nan, np.nan
    pbar = R / N
    T = np.sum(scores * (counts - totals * pbar))
    var = pbar * (1 - pbar) * (np.sum(totals * scores**2) - (np.sum(totals * scores))**2 / N)
    if var <= 0:
        return np.nan, np.nan
    z = T / np.sqrt(var)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return z, p


def tendencia_logistica(df_sub, flag):
    """Regressao logistica flag ~ ano (continuo). Retorna OR/ano, OR/decada, p."""
    d = df_sub.copy()
    d["y"] = d[flag].astype(int)
    d["ano_num"] = d["ano"].astype(int)
    if d["y"].nunique() < 2:
        return {"OR_ano": np.nan, "OR_decada": np.nan, "p": np.nan, "coef": np.nan}
    try:
        m = smf.logit("y ~ ano_num", data=d).fit(disp=0)
        coef = m.params["ano_num"]
        p = m.pvalues["ano_num"]
        return {"OR_ano": float(np.exp(coef)),
                "OR_decada": float(np.exp(coef * 10)),
                "p": float(p), "coef": float(coef)}
    except Exception:
        return {"OR_ano": np.nan, "OR_decada": np.nan, "p": np.nan, "coef": np.nan}

def run_self_tests(df):
    print("\n" + "=" * 70)
    print("AUTO-TESTES (sanidade de parsing e estatistica)")
    print("=" * 70)
    ok = 0; fail = 0
    def check(name, cond):
        nonlocal ok, fail
        status = "PASS" if cond else "FALHOU"
        if cond: ok += 1
        else: fail += 1
        print(f"  [{status}] {name}")

    check("total de alunos == 1920", len(df) == 1920)
    check("Antigo tem 1020 alunos (3 anos)", (df.grupo == "Antigo").sum() == 1020)
    check("Recente tem 900 alunos (3 anos)", (df.grupo == "Recente").sum() == 900)
    check("6 anos presentes", sorted(df.ano.unique()) == ANOS)
    check("3 cursos presentes", sorted(df.curso.unique()) == sorted(CURSOS))
    check("nenhuma inscricao vazia", df.inscricao.str.len().min() > 0)
    test_df = pd.Series(["DA COSTA E SILVA", "ACOSTA JUNIOR", "SOUZA LIMA",
                         "PEREIRA DOS SANTOS", "MARIA DE SOUSA"]).map(norm)
    pat_costa = re.compile(r"(?<![A-Z])COSTA(?![A-Z])")
    pat_souza = re.compile(r"(?<![A-Z])SOUZA(?![A-Z])")
    check("COSTA casa em 'DA COSTA E SILVA'", bool(pat_costa.search(test_df[0])))
    check("COSTA NAO casa em 'ACOSTA' (palavra inteira)", not pat_costa.search(test_df[1]))
    check("SOUZA casa em 'SOUZA LIMA'", bool(pat_souza.search(test_df[2])))
    check("SOUZA NAO casa em 'SOUSA' (grafia diferente)", not pat_souza.search(test_df[4]))
    # Gênero (recomenda-se testar com IA para confirmar o resultado, pois elas têm muito conhecimento global)
    g = {n: infer_sexo(n) for n in ["JOAO", "MARIA", "LUCAS", "BEATRIZ",
                                    "CAUA", "LUCCA", "ALINE", "RAFAEL", "INGRID"]}
    check("genero: MARIA=F", g["MARIA"] == "F")
    check("genero: LUCAS=M", g["LUCAS"] == "M")
    check("genero: BEATRIZ=F", g["BEATRIZ"] == "F")
    check("genero: CAUA=M (excecao termina em a)", g["CAUA"] == "M")
    check("genero: LUCCA=M (excecao)", g["LUCCA"] == "M")
    check("genero: INGRID=F", g["INGRID"] == "F")
    u = (df.sexo == "U").mean()
    check(f"genero desconhecido < 2% (obs={u:.3%})", u < 0.02)
    check("QUALQUER5 == OR dos 5 flags",
          (df["QUALQUER5"] == df[SOBRENOMES].any(axis=1)).all())
    check("soma flags >= casos QUALQUER5",
          df[SOBRENOMES].sum().sum() >= df["QUALQUER5"].sum())

    inesperados = sorted({norm(n).split()[0] for n in df[df.sexo == "U"].nome}
                         - _UNKNOWN)
    if inesperados:
        print(f"  [AVISO] prenomes 'U' fora do dicionario: {inesperados}")

    print(f"\n  Resultado: {ok} passaram, {fail} falharam.")
    return fail == 0

def tabela_por_ano(df):
    """Contagem e proporcao de cada sobrenome (e QUALQUER5) por ano x curso."""
    registros = []
    metricas = SOBRENOMES + ["QUALQUER5"]
    for curso in CURSOS + ["TODOS"]:
        sub_c = df if curso == "TODOS" else df[df.curso == curso]
        for ano in ANOS:
            sub = sub_c[sub_c.ano == ano]
            n = len(sub)
            reg = {"curso": curso, "ano": ano, "grupo": GRUPOS[ano], "n_alunos": n}
            for m in metricas:
                casos = int(sub[m].sum())
                reg[f"{m}_n"] = casos
                reg[f"{m}_%"] = round(100 * casos / n, 2) if n else np.nan
            registros.append(reg)
    return pd.DataFrame(registros)


def analise_grupos(df):
    """Antigo vs Recente para cada sobrenome (e combinado), por curso e TODOS."""
    linhas = []
    metricas = SOBRENOMES + ["QUALQUER5"]
    for curso in CURSOS + ["TODOS"]:
        sub_c = df if curso == "TODOS" else df[df.curso == curso]
        old = sub_c[sub_c.grupo == "Antigo"]
        new = sub_c[sub_c.grupo == "Recente"]
        for m in metricas:
            r = comparar_proporcoes(int(old[m].sum()), len(old),
                                    int(new[m].sum()), len(new))
            linhas.append({
                "curso": curso, "metrica": m,
                "n_antigo": r["n_antigo"], "casos_antigo": r["casos_antigo"],
                "%_antigo": round(100 * r["prop_antigo"], 2),
                "n_recente": r["n_recente"], "casos_recente": r["casos_recente"],
                "%_recente": round(100 * r["prop_recente"], 2),
                "dif_pp": round(r["dif_pp"], 2),
                "variacao_rel_%": round(r["variacao_rel_%"], 1) if r["variacao_rel_%"] == r["variacao_rel_%"] else np.nan,
                "direcao": r["direcao"],
                "RR": round(r["RR"], 3), "RR_ic95": f"[{r['RR_ic'][0]:.2f}; {r['RR_ic'][1]:.2f}]",
                "OR": round(r["OR"], 3), "OR_ic95": f"[{r['OR_ic'][0]:.2f}; {r['OR_ic'][1]:.2f}]",
                "teste": r["teste_principal"],
                "p_valor": round(r["p_principal"], 4),
                "p_ztest": round(r["p_ztest"], 4),
                "significativo": r["sig"],
            })
    return pd.DataFrame(linhas)


def analise_tendencia(df):
    """Tendencia  6 anos (Cochran-Armitage + logistica)."""
    linhas = []
    metricas = SOBRENOMES + ["QUALQUER5"]
    for curso in CURSOS + ["TODOS"]:
        sub_c = df if curso == "TODOS" else df[df.curso == curso]
        for m in metricas:
            counts, totals, scores = [], [], []
            for ano in ANOS:
                s = sub_c[sub_c.ano == ano]
                counts.append(int(s[m].sum())); totals.append(len(s)); scores.append(int(ano))
            z, p = cochran_armitage(counts, totals, scores)
            logit = tendencia_logistica(sub_c, m)
            linhas.append({
                "curso": curso, "metrica": m,
                "CA_z": round(z, 3) if z == z else np.nan,
                "CA_p": round(p, 4) if p == p else np.nan,
                "logit_OR_por_ano": round(logit["OR_ano"], 4) if logit["OR_ano"] == logit["OR_ano"] else np.nan,
                "logit_OR_por_decada": round(logit["OR_decada"], 3) if logit["OR_decada"] == logit["OR_decada"] else np.nan,
                "logit_p": round(logit["p"], 4) if logit["p"] == logit["p"] else np.nan,
                "tendencia": ("subindo" if logit["coef"] > 0 else "descendo") if logit["coef"] == logit["coef"] else "n/d",
            })
    return pd.DataFrame(linhas)


def analise_sexo(df):
    """SOBRENOMES por SEXO: o desfecho e SEMPRE 'tem um dos sobrenomes'.
    Compara a PROPORCAO de portadores do sobrenome entre MULHERES e HOMENS,
    por curso e por grupo. Tambem compara essa proporcao ENTRE CURSOS, em cada
    sexo. Nada aqui e contagem bruta: tudo e proporcao (%)."""
    dfg = df[df.sexo != "U"].copy()  
    metricas = SOBRENOMES + ["QUALQUER5"]
    tabelas = {}

    # homem e mulher = Agora corrigido para a proporcao de cada sexo, e não a contagem bruta com fiz antes, que burro ]:
    reg = []
    for curso in CURSOS + ["TODOS"]:
        sc = dfg if curso == "TODOS" else dfg[dfg.curso == curso]
        for grupo in ["Ambos", "Antigo", "Recente"]:
            sg = sc if grupo == "Ambos" else sc[sc.grupo == grupo]
            F = sg[sg.sexo == "F"]; M = sg[sg.sexo == "M"]
            for m in metricas:
                r = comparar_proporcoes(int(M[m].sum()), len(M),
                                        int(F[m].sum()), len(F))
                reg.append({
                    "curso": curso, "grupo": grupo, "sobrenome": m,
                    "n_homens": len(M), "casos_H": int(M[m].sum()),
                    "%_homens": round(100 * r["prop_antigo"], 2),
                    "n_mulheres": len(F), "casos_M": int(F[m].sum()),
                    "%_mulheres": round(100 * r["prop_recente"], 2),
                    "dif_pp_(M-H)": round(r["dif_pp"], 2),
                    "maior_em": "MULHERES" if r["dif_pp"] > 0 else ("HOMENS" if r["dif_pp"] < 0 else "igual"),
                    "OR_(M/H)": round(r["OR"], 3),
                    "OR_ic95": f"[{r['OR_ic'][0]:.2f}; {r['OR_ic'][1]:.2f}]",
                    "teste": r["teste_principal"],
                    "p_valor": round(r["p_principal"], 4),
                    "significativo": r["sig"],
                })
    tabelas["sobrenome_por_sexo"] = pd.DataFrame(reg)

    
    reg = []
    for sexo, lab in [("F", "MULHERES"), ("M", "HOMENS")]:
        ss = dfg[dfg.sexo == sexo]
        for grupo in ["Ambos", "Antigo", "Recente"]:
            sg = ss if grupo == "Ambos" else ss[ss.grupo == grupo]
            linha = {"sexo": lab, "grupo": grupo}
            tab = []
            for curso in CURSOS:
                s = sg[sg.curso == curso]
                casos = int(s["QUALQUER5"].sum()); n = len(s)
                linha[f"%_{curso.split()[0][:4]}"] = round(100 * casos / n, 2) if n else np.nan
                tab.append([casos, n - casos])
            tab = np.array(tab)
            try:
                chi2, p, dof, exp = stats.chi2_contingency(tab)
                if exp.min() < 5:
                    p = stats.chi2_contingency(tab)[1]  # mantem; aviso abaixo
                linha["qui2"] = round(chi2, 2); linha["p_entre_cursos"] = round(p, 4)
                linha["esperada_min"] = round(exp.min(), 1)
            except Exception:
                linha["qui2"] = np.nan; linha["p_entre_cursos"] = np.nan
            reg.append(linha)
    tabelas["sobrenome_entre_cursos_por_sexo"] = pd.DataFrame(reg)

    
    reg = []
    for sexo, lab in [("F", "MULHERES"), ("M", "HOMENS")]:
        ss = dfg[dfg.sexo == sexo]
        for curso in CURSOS + ["TODOS"]:
            sc = ss if curso == "TODOS" else ss[ss.curso == curso]
            counts, totals, scores = [], [], []
            for ano in ANOS:
                s = sc[sc.ano == ano]
                counts.append(int(s["QUALQUER5"].sum())); totals.append(len(s)); scores.append(int(ano))
            z, p = cochran_armitage(counts, totals, scores)
            reg.append({"sexo": lab, "curso": curso,
                        "%_antigo": round(100 * sum(counts[:3]) / max(sum(totals[:3]), 1), 2),
                        "%_recente": round(100 * sum(counts[3:]) / max(sum(totals[3:]), 1), 2),
                        "CA_p": round(p, 4) if p == p else np.nan})
    tabelas["sobrenome_sexo_tendencia"] = pd.DataFrame(reg)

    return tabelas


def analise_sexo_periodo(df):
    """O CORE do estudo de cotas: para cada SEXO e CURSO, a proporcao dos 5
    sobrenomes em 2003-05 (Antigo) vs 2023-25 (Recente), com teste da diferenca.
    Inclui cada sobrenome e os 5 combinados. Tudo em proporcao (%)."""
    dfg = df[df.sexo != "U"].copy()
    metricas = SOBRENOMES + ["QUALQUER5"]
    linhas = []
    for sexo, lab in [("ALL", "Todos os sexos"), ("F", "Mulheres"), ("M", "Homens")]:
        ss = dfg if sexo == "ALL" else dfg[dfg.sexo == sexo]
        for curso in CURSOS + ["TODOS"]:
            sc = ss if curso == "TODOS" else ss[ss.curso == curso]
            old = sc[sc.grupo == "Antigo"]; new = sc[sc.grupo == "Recente"]
            for m in metricas:
                r = comparar_proporcoes(int(old[m].sum()), len(old),
                                        int(new[m].sum()), len(new))
                linhas.append({
                    "sexo": lab, "curso": curso, "sobrenome": m,
                    "n_2003_05": r["n_antigo"], "casos_2003_05": r["casos_antigo"],
                    "%_2003_05": round(100 * r["prop_antigo"], 2),
                    "n_2023_25": r["n_recente"], "casos_2023_25": r["casos_recente"],
                    "%_2023_25": round(100 * r["prop_recente"], 2),
                    "dif_pp": round(r["dif_pp"], 2), "direcao": r["direcao"],
                    "OR": round(r["OR"], 3), "OR_ic95": f"[{r['OR_ic'][0]:.2f}; {r['OR_ic'][1]:.2f}]",
                    "teste": r["teste_principal"], "p_valor": round(r["p_principal"], 4),
                    "significativo": r["sig"],
                })
    return pd.DataFrame(linhas)


def imprimir_relatorio(df, t_ano, t_grupo, t_trend, t_sexo, t_sxp):
    L = []
    p = L.append
    p("=" * 78)
    p("RELATORIO - 5 SOBRENOMES (Silva, Santos, Oliveira, Souza, Pereira)")
    p("Efeito das politicas de cotas: UFSC 2003-05 (pre) vs 2023-25 (pos)")
    p("=" * 78)
    p("NOTA: 'QUALQUER5' = aluno que tem PELO MENOS UM dos 5 sobrenomes.")
    p("      'TODOS' = todos os cursos somados (Arquitetura+Direito+Medicina).")
    p(f"Total: {len(df)} alunos | 2003-05: {(df.grupo=='Antigo').sum()} | "
      f"2023-25: {(df.grupo=='Recente').sum()}")
    nu = (df.sexo == "U").sum()
    p(f"Sexo: F={(df.sexo=='F').sum()}  M={(df.sexo=='M').sum()}  Desconhecido(U)={nu}")
    p("")

    p("-" * 78)
    p("1) 2003-05 vs 2023-25 - cada sobrenome e os 5 combinados (todos os cursos)")
    p("-" * 78)
    sub = t_grupo[t_grupo.curso == "TODOS"]
    for _, r in sub.iterrows():
        p(f"  {r['metrica']:<10} {r['%_antigo']:>6.2f}% -> {r['%_recente']:>6.2f}%"
          f"  ({r['dif_pp']:+.2f} pp, {r['direcao']:<8})  OR={r['OR']:.2f} {r['OR_ic95']}"
          f"  p={r['p_valor']:.4f} [{r['significativo']}]")
    p("")

    p("-" * 78)
    p("2) 2003-05 vs 2023-25 por CURSO (5 sobrenomes combinados - QUALQUER5)")
    p("-" * 78)
    sub = t_grupo[t_grupo.metrica == "QUALQUER5"]
    for _, r in sub.iterrows():
        nome = "Todos os cursos" if r['curso'] == "TODOS" else r['curso']
        p(f"  {nome:<24} {r['%_antigo']:>6.2f}% -> {r['%_recente']:>6.2f}%"
          f"  ({r['dif_pp']:+.2f} pp, {r['direcao']:<8})  p={r['p_valor']:.4f} [{r['significativo']}]")
    p("")

    p("-" * 78)
    p("3) TENDENCIA ao longo dos 6 anos (todos os cursos)")
    p("-" * 78)
    sub = t_trend[t_trend.curso == "TODOS"]
    for _, r in sub.iterrows():
        p(f"  {r['metrica']:<10} Cochran-Armitage p={r['CA_p']:.4f}"
          f"   logit OR/decada={r['logit_OR_por_decada']:.2f}  p={r['logit_p']:.4f}  ({r['tendencia']})")
    p("")

    p("-" * 78)
    p("4) POR SEXO - 5 sobrenomes (QUALQUER5): 2003-05 vs 2023-25, por curso")
    p("   (cada sexo tem o periodo antigo E o recente, com teste da diferenca)")
    p("-" * 78)
    sub = t_sxp[t_sxp.sobrenome == "QUALQUER5"]
    for sexo in ["Mulheres", "Homens"]:
        p(f"  [{sexo}]")
        for _, r in sub[sub.sexo == sexo].iterrows():
            nome = "Todos os cursos" if r['curso'] == "TODOS" else r['curso']
            p(f"    {nome:<24} {r['%_2003_05']:>6.2f}% -> {r['%_2023_25']:>6.2f}%"
              f"  ({r['dif_pp']:+.2f} pp, {r['direcao']:<8})  OR={r['OR']:.2f}"
              f"  p={r['p_valor']:.4f} [{r['significativo']}]")
    p("")
    txt = "\n".join(L)
    print(txt)
    return txt

#Gráfico (recomendo solicitar melhoria em IA porque o matplotlib é muito limitado; o seaborn é até bom, mas IA é melhor)
def gerar_graficos(df, t_ano, t_grupo, t_sxp, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy.stats import gaussian_kde

    C = {
        "antigo":  "#3D5A80",   # azul-ardosia (2003-05)
        "recente": "#EE6C4D",   # terracota (2023-25)
        "fem":     "#C44B8B",   # rosa-magenta (meninas)
        "masc":    "#1B998B",   # verde-azulado (meninos)
        "geral":   "#293241",   # grafite (geral)
        "arq":     "#8D99AE",
        "dir":     "#EE6C4D",
        "med":     "#3D5A80",
        "todos":   "#293241",
    }
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
        "figure.dpi": 130,
    })
    curso_cor = {"ARQUITETURA E URBANISMO": C["arq"], "DIREITO": C["dir"],
                 "MEDICINA": C["med"], "TODOS": C["todos"]}
    curso_lbl = {"ARQUITETURA E URBANISMO": "Arquitetura", "DIREITO": "Direito",
                 "MEDICINA": "Medicina", "TODOS": "Todos os cursos"}

    def stars(pv):
        return "***" if pv < .001 else "**" if pv < .01 else "*" if pv < .05 else "n.s."

    def wilson(c, n):
        lo, hi = proportion_confint(c, n, method="wilson")
        return 100 * c / n, 100 * lo, 100 * hi

    paths = []

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    xpos = {a: i for i, a in enumerate(ANOS)}
    for curso in CURSOS + ["TODOS"]:
        s = t_ano[t_ano.curso == curso].sort_values("ano")
        axL.plot([xpos[a] for a in s.ano], s["QUALQUER5_%"], marker="o", ms=6,
                 lw=2.2, color=curso_cor[curso], label=curso_lbl[curso],
                 zorder=3 if curso == "TODOS" else 2)
    axL.axvspan(-0.4, 2.4, color=C["antigo"], alpha=0.07)
    axL.axvspan(2.6, 5.4, color=C["recente"], alpha=0.07)
    axL.text(1, axL.get_ylim()[1]*0.97, "PRE-cotas\n2003-05", ha="center",
             va="top", fontsize=8.5, color=C["antigo"], weight="bold")
    axL.text(4, axL.get_ylim()[1]*0.97, "POS-cotas\n2023-25", ha="center",
             va="top", fontsize=8.5, color=C["recente"], weight="bold")
    axL.set_xticks(range(6)); axL.set_xticklabels(ANOS)
    axL.set_ylabel("% de alunos com algum dos 5 sobrenomes")
    axL.set_title("(A) Evolucao por ano e curso")
    axL.legend(fontsize=8.5, frameon=False, loc="upper left")

    cursos = CURSOS + ["TODOS"]
    x = np.arange(len(cursos)); w = 0.38
    for k, (grp, cor, off) in enumerate([("Antigo", C["antigo"], -w/2),
                                         ("Recente", C["recente"], +w/2)]):
        vals, los, his = [], [], []
        for curso in cursos:
            sc = df if curso == "TODOS" else df[df.curso == curso]
            sg = sc[sc.grupo == grp]
            v, lo, hi = wilson(int(sg["QUALQUER5"].sum()), len(sg))
            vals.append(v); los.append(v-lo); his.append(hi-v)
        axR.bar(x+off, vals, w, color=cor, yerr=[los, his], capsize=3,
                ecolor="#555", error_kw={"lw": 1},
                label="2003-05 (pre)" if grp == "Antigo" else "2023-25 (pos)")
    sub = t_grupo[t_grupo.metrica == "QUALQUER5"].set_index("curso")
    for i, curso in enumerate(cursos):
        pv = sub.loc[curso, "p_valor"]
        ytop = max(sub.loc[curso, "%_antigo"], sub.loc[curso, "%_recente"]) + 4
        axR.text(i, ytop, stars(pv), ha="center", fontsize=11, weight="bold")
    axR.set_xticks(x); axR.set_xticklabels([curso_lbl[c] for c in cursos], rotation=12)
    axR.set_ylabel("% com algum dos 5 sobrenomes")
    axR.set_title("(B) Pre vs pos-cotas (IC 95%)")
    axR.legend(fontsize=8.5, frameon=False)
    fig.suptitle("Os 5 sobrenomes mais comuns como proxy do efeito das cotas (UFSC)",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.text(0.5, -0.04, "* p<0,05   ** p<0,01   *** p<0,001   n.s. = nao significativo",
             ha="center", fontsize=8.5, color="#555")
    fig.tight_layout()
    f = os.path.join(outdir, "fig1_panorama.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig); paths.append(f)

 
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    dfg = df[df.sexo != "U"]
    grupos_sexo = [("Mulheres", "F", C["fem"]), ("Homens", "M", C["masc"]),
                   ("Todos os\nsexos", "ALL", C["geral"])]
    x = np.arange(len(grupos_sexo)); w = 0.38
    for grp, cor, off, hatch in [("Antigo", None, -w/2, ""), ("Recente", None, +w/2, "//")]:
        vals, los, his = [], [], []
        for lab, sx, base in grupos_sexo:
            ss = dfg if sx == "ALL" else dfg[dfg.sexo == sx]
            sg = ss[ss.grupo == grp]
            v, lo, hi = wilson(int(sg["QUALQUER5"].sum()), len(sg))
            vals.append(v); los.append(v-lo); his.append(hi-v)
        cores = [g[2] for g in grupos_sexo]
        axL.bar(x+off, vals, w, color=cores, alpha=0.55 if grp == "Antigo" else 1.0,
                hatch=hatch, edgecolor="white", yerr=[los, his], capsize=3,
                ecolor="#555", error_kw={"lw": 1})
    sub = t_sxp[(t_sxp.sobrenome == "QUALQUER5") & (t_sxp.curso == "TODOS")].set_index("sexo")
    for i, (lab, sx, base) in enumerate(grupos_sexo):
        key = {"F": "Mulheres", "M": "Homens", "ALL": "Todos os sexos"}[sx]
        pv = sub.loc[key, "p_valor"]
        ytop = max(sub.loc[key, "%_2003_05"], sub.loc[key, "%_2023_25"]) + 3
        axL.text(i, ytop, stars(pv), ha="center", fontsize=11, weight="bold")
    axL.set_xticks(x); axL.set_xticklabels([g[0] for g in grupos_sexo])
    axL.set_ylabel("% com algum dos 5 sobrenomes")
    axL.set_title("(A) Pre vs pos-cotas por sexo (todos os cursos)")
    axL.legend(handles=[Patch(facecolor="#999", alpha=0.55, label="2003-05 (pre)"),
                        Patch(facecolor="#999", hatch="//", label="2023-25 (pos)")],
               fontsize=8.5, frameon=False)

    rows = []
    for curso in CURSOS:
        for sx, cor in [("F", C["fem"]), ("M", C["masc"])]:
            r = t_sxp[(t_sxp.sobrenome == "QUALQUER5") & (t_sxp.curso == curso) &
                      (t_sxp.sexo == {"F": "Mulheres", "M": "Homens"}[sx])].iloc[0]
            rows.append((f"{curso_lbl[curso]}\n{'Mulheres' if sx=='F' else 'Homens'}",
                         r["%_2003_05"], r["%_2023_25"], cor, r["p_valor"]))
    rows = rows[::-1]
    for j, (lab, a, b, cor, pv) in enumerate(rows):
        axR.plot([a, b], [j, j], color=cor, lw=2.5, zorder=1, alpha=0.6)
        axR.scatter(a, j, s=70, color="white", edgecolor=cor, lw=2, zorder=2)
        axR.scatter(b, j, s=90, color=cor, zorder=3)
        axR.text(b + 0.6, j, f"{b-a:+.1f} pp {stars(pv)}", va="center", fontsize=8)
    axR.set_yticks(range(len(rows))); axR.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    axR.set_xlabel("% com algum dos 5 sobrenomes")
    axR.set_title("(B) Salto pre -> pos (anel=2003-05, cheio=2023-25)")
    axR.scatter([], [], color=C["fem"], label="Mulheres")
    axR.scatter([], [], color=C["masc"], label="Homens")
    axR.legend(fontsize=8.5, frameon=False, loc="lower right")
    fig.suptitle("Recorte por sexo: a mudanca entre os dois periodos",
                 fontsize=13.5, weight="bold", y=1.02)
    fig.tight_layout()
    f = os.path.join(outdir, "fig2_por_sexo.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig); paths.append(f)


    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2))
    anos_i = range(6)
    fem_c, masc_c, tot_geral = [], [], []
    for a in ANOS:
        s = df[df.ano == a]
        port = s[s["QUALQUER5"]]
        fem_c.append((port.sexo == "F").sum())
        masc_c.append((port.sexo == "M").sum())
        tot_geral.append(len(s))
    fem_c = np.array(fem_c); masc_c = np.array(masc_c); tot_geral = np.array(tot_geral)
    axL.fill_between(anos_i, 0, masc_c, color=C["masc"], alpha=0.85, label="Meninos com sobrenome")
    axL.fill_between(anos_i, masc_c, masc_c+fem_c, color=C["fem"], alpha=0.85, label="Meninas com sobrenome")
    axL.plot(anos_i, masc_c+fem_c, color=C["geral"], lw=2, marker="o", ms=4,
             label="Total com algum dos 5")
    ax2 = axL.twinx()
    ax2.plot(anos_i, 100*(masc_c+fem_c)/tot_geral, color="#E0A500", lw=2.2,
             ls="--", marker="s", ms=5, label="% sobre o total de alunos")
    ax2.set_ylabel("% sobre o total de alunos do ano", color="#B58300")
    ax2.tick_params(axis="y", colors="#B58300"); ax2.grid(False)
    axL.axvspan(-0.3, 2.3, color=C["antigo"], alpha=0.07)
    axL.axvspan(2.7, 5.3, color=C["recente"], alpha=0.07)
    axL.set_xticks(list(anos_i)); axL.set_xticklabels(ANOS)
    axL.set_ylabel("Numero de alunos com algum dos 5 sobrenomes")
    axL.set_title("(A) Quantidade e composicao por sexo (com % geral sobreposta)")
    h1, l1 = axL.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    axL.legend(h1+h2, l1+l2, fontsize=8, frameon=False, loc="upper left")

    def props_cells(grupo):
        vals = []
        sub = df[df.grupo == grupo]
        for curso in CURSOS:
            for a in ANOS:
                s = sub[(sub.curso == curso) & (sub.ano == a)]
                if len(s):
                    vals.append(100 * s["QUALQUER5"].mean())
        return np.array(vals)
    pa = props_cells("Antigo"); pr = props_cells("Recente")
    grid = np.linspace(0, max(pa.max(), pr.max())*1.25, 200)
    for vals, cor, lab in [(pa, C["antigo"], "2003-05 (pre)"), (pr, C["recente"], "2023-25 (pos)")]:
        kde = gaussian_kde(vals)
        axR.fill_between(grid, kde(grid), color=cor, alpha=0.30)
        axR.plot(grid, kde(grid), color=cor, lw=2.2, label=lab)
        axR.axvline(vals.mean(), color=cor, ls="--", lw=1.5)
        axR.scatter(vals, np.full_like(vals, -kde(grid).max()*0.03), color=cor,
                    marker="|", s=120, alpha=0.7)
    axR.set_xlabel("% com algum dos 5 sobrenomes (por celula curso x ano)")
    axR.set_ylabel("Densidade")
    axR.set_title("(B) Distribuicao deslocou para a direita (linha tracejada = media)")
    axR.legend(fontsize=9, frameon=False)
    fig.suptitle("Densidade e composicao: quantos sao, quem sao, e como a distribuicao mudou",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    f = os.path.join(outdir, "fig3_densidade.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig); paths.append(f)

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2),
                                   gridspec_kw={"width_ratios": [1, 1]})
    sub = t_grupo[(t_grupo.curso == "TODOS")].set_index("metrica")
    nomes = SOBRENOMES
    x = np.arange(len(nomes)); w = 0.38
    axL.bar(x-w/2, [sub.loc[m, "%_antigo"] for m in nomes], w, color=C["antigo"], label="2003-05 (pre)")
    axL.bar(x+w/2, [sub.loc[m, "%_recente"] for m in nomes], w, color=C["recente"], label="2023-25 (pos)")
    for i, m in enumerate(nomes):
        ytop = max(sub.loc[m, "%_antigo"], sub.loc[m, "%_recente"]) + 0.4
        axL.text(i, ytop, stars(sub.loc[m, "p_valor"]), ha="center", fontsize=10, weight="bold")
    axL.set_xticks(x); axL.set_xticklabels([n.title() for n in nomes], rotation=12)
    axL.set_ylabel("% dos alunos"); axL.set_title("(A) Cada sobrenome: pre vs pos")
    axL.legend(fontsize=8.5, frameon=False)

    metr = SOBRENOMES + ["QUALQUER5"]
    ors, los, his, labs, cols = [], [], [], [], []
    for m in metr:
        r = sub.loc[m]
        ors.append(r["OR"]); 
        lo, hi = [float(v) for v in r["OR_ic95"].strip("[]").split(";")]
        los.append(lo); his.append(hi)
        labs.append(("Qualquer dos 5" if m == "QUALQUER5" else m.title()))
        cols.append(C["recente"] if r["significativo"] == "SIM" else "#9aa0a6")
    ypos = np.arange(len(metr))[::-1]
    for y, o, lo, hi, c in zip(ypos, ors, los, his, cols):
        axR.plot([lo, hi], [y, y], color=c, lw=2.2)
        axR.scatter(o, y, s=80, color=c, zorder=3)
    axR.axvline(1, color="#293241", ls="--", lw=1.2)
    axR.set_yticks(ypos); axR.set_yticklabels(labs)
    axR.set_xlabel("Razao de chances (OR) pos vs pre  [IC 95%]")
    axR.set_title("(B) Tamanho de efeito (laranja = significativo)")
    axR.set_xscale("log")
    from matplotlib.ticker import NullLocator, FixedLocator, FixedFormatter
    axR.xaxis.set_minor_locator(NullLocator())
    axR.xaxis.set_major_locator(FixedLocator([0.5, 1, 2, 4]))
    axR.xaxis.set_major_formatter(FixedFormatter(["0,5", "1", "2", "4"]))
    fig.suptitle("Sobrenome a sobrenome: quem puxa o aumento e o tamanho do efeito",
                 fontsize=13, weight="bold", y=1.02)
    fig.tight_layout()
    f = os.path.join(outdir, "fig4_por_sobrenome.png")
    fig.savefig(f, bbox_inches="tight"); plt.close(fig); paths.append(f)

    return paths


def exportar_excel(df, t_ano, t_grupo, t_trend, t_sexo, t_sxp, relatorio, path):
    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        pd.DataFrame({"Relatorio": relatorio.split("\n")}).to_excel(
            xw, sheet_name="Resumo", index=False)
        t_grupo.to_excel(xw, sheet_name="Pre_vs_Pos", index=False)
        t_sxp.to_excel(xw, sheet_name="Por_sexo_e_periodo", index=False)
        t_ano.to_excel(xw, sheet_name="Por_ano", index=False)
        t_trend.to_excel(xw, sheet_name="Tendencia", index=False)
        t_sexo["sobrenome_por_sexo"].to_excel(xw, sheet_name="Sobrenome_x_sexo", index=False)
        t_sexo["sobrenome_entre_cursos_por_sexo"].to_excel(xw, sheet_name="Entre_cursos_x_sexo", index=False)
        t_sexo["sobrenome_sexo_tendencia"].to_excel(xw, sheet_name="Tendencia_x_sexo", index=False)
        df.drop(columns=["nome_norm"]).to_excel(xw, sheet_name="Dados_brutos", index=False)
    return path


def main():
    df = load_data()
    run_self_tests(df)
    t_ano = tabela_por_ano(df)
    t_grupo = analise_grupos(df)
    t_trend = analise_tendencia(df)
    t_sexo = analise_sexo(df)
    t_sxp = analise_sexo_periodo(df)
    relatorio = imprimir_relatorio(df, t_ano, t_grupo, t_trend, t_sexo, t_sxp)
    graficos = gerar_graficos(df, t_ano, t_grupo, t_sxp, OUTDIR)
    xlsx = exportar_excel(df, t_ano, t_grupo, t_trend, t_sexo, t_sxp, relatorio,
                          os.path.join(OUTDIR, "analise_sobrenomes_resultados.xlsx"))
    print("\nArquivos gerados:")
    print(" -", xlsx)
    for g in graficos:
        print(" -", g)
    return df, t_ano, t_grupo, t_trend, t_sexo, t_sxp


if __name__ == "__main__":
    main()
