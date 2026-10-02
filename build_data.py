"""
builds site_data.json: predictions + the full math + h2h + form + insights
for the next matchday in 8 european leagues, plus receipts (how past picks did),
season projections, and bookmaker odds when football-data.co.uk is reachable.

data: openfootball/football.json on github (results back to 2010-11)
run:  python build_data.py   (downloads the data itself)
"""

import glob
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

HALF_LIFE_DAYS = 240
PENALTY = 1.5        # how hard ratings get pulled toward their prior (bigger = more cautious)
GRID = 7  # show 0..6 goals in the score grid
CURRENT = "2026-27"
LAST = "2025-26"
PREV = "2024-25"
FIRST_YEAR = 2010
URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"

LEAGUES = [
    # code, name, country, football-data code, table zones
    ("en.1", "Premier League", "England", "E0", [("title", "Title", 1, 1), ("top", "Top 4", 1, 4), ("down", "Relegated", -3, -1)]),
    ("es.1", "La Liga", "Spain", "SP1", [("title", "Title", 1, 1), ("top", "Top 4", 1, 4), ("down", "Relegated", -3, -1)]),
    ("de.1", "Bundesliga", "Germany", "D1", [("title", "Title", 1, 1), ("top", "Top 4", 1, 4), ("down", "Bottom 2", -2, -1)]),
    ("it.1", "Serie A", "Italy", "I1", [("title", "Title", 1, 1), ("top", "Top 4", 1, 4), ("down", "Relegated", -3, -1)]),
    ("fr.1", "Ligue 1", "France", "F1", [("title", "Title", 1, 1), ("top", "Top 3", 1, 3), ("down", "Bottom 2", -2, -1)]),
    ("en.2", "Championship", "England", "E1", [("title", "Top 2", 1, 2), ("top", "Top 6", 1, 6), ("down", "Relegated", -3, -1)]),
    ("nl.1", "Eredivisie", "Netherlands", "N1", [("title", "Title", 1, 1), ("top", "Top 2", 1, 2), ("down", "Bottom 2", -2, -1)]),
    ("pt.1", "Primeira Liga", "Portugal", "P1", [("title", "Title", 1, 1), ("top", "Top 2", 1, 2), ("down", "Bottom 2", -2, -1)]),
]


def zone_list(z):
    # positive = from the top, negative = from the bottom (-3..-1 = last three)
    return [{"key": k, "label": l, "from": a, "to": b} for k, l, a, b in z]


def clean(name):
    for junk in (" AFC", " FC"):
        if name.endswith(junk):
            name = name[: -len(junk)]
    if name.startswith("AFC "):
        name = name[4:]
    return name.strip()


def download(code):
    """current + last season always refreshed, older seasons only fetched once."""
    import requests
    os.makedirs(f"data2/{code}", exist_ok=True)
    cur_year = int(CURRENT[:4])
    for y in range(FIRST_YEAR, cur_year + 1):
        season = f"{y}-{(y + 1) % 100:02d}"
        path = f"data2/{code}/{season}.json"
        if os.path.exists(path) and season not in (CURRENT, LAST):
            continue
        r = requests.get(URL.format(season=season, code=code), timeout=30)
        if r.status_code == 200:
            open(path, "w").write(r.text)


def load(code):
    rows = []
    for f in sorted(glob.glob(f"data2/{code}/*.json")):
        season = os.path.basename(f)[:-5]
        for m in json.load(open(f))["matches"]:
            s = m.get("score")
            if isinstance(s, list):
                ft, ht = s, None
            else:
                ft, ht = (s or {}).get("ft"), (s or {}).get("ht")
            rows.append(dict(
                season=season, date=pd.to_datetime(m["date"]), time=m.get("time", ""),
                round=m.get("round", ""), home=clean(m["team1"]), away=clean(m["team2"]),
                hg=ft[0] if ft else np.nan, ag=ft[1] if ft else np.nan,
                hht=ht[0] if ht else np.nan, aht=ht[1] if ht else np.nan,
            ))
    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)


# ---------------- model ----------------
#
# goals ~ Poisson.  log(home rate) = mu + div + attack[home] - defense[away] + home_adv
#                   log(away rate) = mu + div + attack[away] - defense[home]
# + Dixon-Coles tweak for the low scores (0-0, 1-0, 0-1, 1-1) that plain Poisson gets wrong.
# England's two leagues are fit together so promoted / relegated teams carry their history over.
# teams new to a league with no history start from a "promoted team" prior instead of average.

JOINT = {"en.1": ["en.1", "en.2"], "en.2": ["en.1", "en.2"]}
PROMOTED_PRIOR = -0.15     # promoted sides start a bit below average in attack and defense
_fit_cache = {}


def training_rows(frames, code, seasons, cutoff):
    codes = JOINT.get(code, [code])
    parts = []
    for i, c in enumerate(codes):
        f = frames[c]
        parts.append(f[f.season.isin(seasons) & (f.date < cutoff)].dropna(subset=["hg"]).assign(div=i))
    return pd.concat(parts), codes.index(code), codes


def promoted_teams(frames, code, season, prev_season):
    """teams in this league this season with no games in the league family last season."""
    codes = JOINT.get(code, [code])
    now = set(frames[code][frames[code].season == season].home)
    before = set()
    for c in codes:
        f = frames[c]
        before |= set(f[f.season == prev_season].home) | set(f[f.season == prev_season].away)
    return now - before


def fit(rows, today, promoted=()):
    key = (len(rows), str(rows.date.max()), str(today), tuple(sorted(promoted)), int(rows["div"].max()))
    if key in _fit_cache:
        return _fit_cache[key]
    teams = sorted(set(rows.home) | set(rows.away) | set(promoted))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h = rows.home.map(idx).values
    a = rows.away.map(idx).values
    hg = rows.hg.values.astype(float)
    ag = rows.ag.values.astype(float)
    dv = rows["div"].values.astype(float)
    has_div = dv.max() > 0
    w = 0.5 ** ((today - rows.date).dt.days.values / HALF_LIFE_DAYS)
    center = np.array([PROMOTED_PRIOR if t in promoted else 0.0 for t in teams])

    # params: att[n], def[n], home_adv, mu, div_offset
    def f_and_grad(p):
        att, dfn = p[:n], p[n:2 * n]
        ha, mu, d = p[2 * n], p[2 * n + 1], p[2 * n + 2]
        eta_h = mu + d * dv + att[h] - dfn[a] + ha
        eta_a = mu + d * dv + att[a] - dfn[h]
        lh, la = np.exp(eta_h), np.exp(eta_a)
        nll = -np.sum(w * (hg * eta_h - lh + ag * eta_a - la))
        nll += PENALTY * (np.sum((att - center) ** 2) + np.sum((dfn - center) ** 2))
        gh, ga = w * (hg - lh), w * (ag - la)
        g = np.zeros_like(p)
        g[:n] = -(np.bincount(h, gh, n) + np.bincount(a, ga, n)) + 2 * PENALTY * (att - center)
        g[n:2 * n] = (np.bincount(a, gh, n) + np.bincount(h, ga, n)) + 2 * PENALTY * (dfn - center)
        g[2 * n] = -gh.sum()
        g[2 * n + 1] = -(gh.sum() + ga.sum())
        g[2 * n + 2] = -(gh * dv).sum() - (ga * dv).sum() if has_div else 0.0
        return nll, g

    x0 = np.zeros(2 * n + 3)
    x0[2 * n] = 0.2
    x0[2 * n + 1] = np.log(max(np.r_[hg, ag].mean(), 0.5))
    p = minimize(f_and_grad, x0, jac=True, method="L-BFGS-B").x
    att, dfn = p[:n], p[n:2 * n]
    ha, mu, d = p[2 * n], p[2 * n + 1], p[2 * n + 2]

    # dixon-coles rho from the fitted rates
    lh = np.exp(mu + d * dv + att[h] - dfn[a] + ha)
    la = np.exp(mu + d * dv + att[a] - dfn[h])
    low = (hg <= 1) & (ag <= 1)

    def neg_tau(rho):
        t = tau(hg[low], ag[low], lh[low], la[low], rho)
        if np.any(t <= 0):
            return 1e9
        return -np.sum(w[low] * np.log(t))

    rho = float(minimize_scalar(neg_tau, bounds=(-0.25, 0.25), method="bounded").x)
    out = dict(teams=teams, att=pd.Series(att, teams), dfn=pd.Series(dfn, teams),
               ha=float(ha), mu=float(mu), d=float(d), rho=rho)
    _fit_cache[key] = out
    return out


def tau(x, y, lh, la, rho):
    t = np.ones_like(lh, dtype=float)
    t = np.where((x == 0) & (y == 0), 1 - lh * la * rho, t)
    t = np.where((x == 0) & (y == 1), 1 + lh * rho, t)
    t = np.where((x == 1) & (y == 0), 1 + la * rho, t)
    t = np.where((x == 1) & (y == 1), 1 - rho, t)
    return t


def league_view(m, league_teams, div):
    """ratings centred on this league's teams, plus the neutral-ground base rate."""
    teams = [t for t in league_teams if t in m["att"].index]
    A, D = m["att"][teams].mean(), m["dfn"][teams].mean()
    r = pd.DataFrame({"attack": m["att"][teams] - A, "defense": m["dfn"][teams] - D})
    base = m["mu"] + m["d"] * div + A - D + m["ha"] / 2
    return r, m["ha"], float(base), m["rho"]


def predict(home, away, r, ha, base, rho=0.0, grid_n=11):
    lh = np.exp(base + r.attack[home] - r.defense[away] + ha / 2)
    la = np.exp(base + r.attack[away] - r.defense[home] - ha / 2)
    g = np.arange(grid_n)
    grid = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    for (x, y) in ((0, 0), (0, 1), (1, 0), (1, 1)):
        grid[x, y] *= tau(np.array(x), np.array(y), np.array(lh), np.array(la), rho)
    grid /= grid.sum()
    return lh, la, grid


def outcome_probs(grid):
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()


def model_for(frames, code, season, prev, cutoff, league_teams):
    rows, div, _ = training_rows(frames, code, [prev, season], cutoff)
    promo = promoted_teams(frames, code, season, prev)
    m = fit(rows, cutoff, promo)
    return league_view(m, league_teams, div), len(rows[rows["div"] == div])


# ---------------- walk forward: receipts (this season) + track record (last season) ----------------

def walk_forward(frames, code, season, prev):
    df = frames[code]
    played = df[(df.season == season)].dropna(subset=["hg"])
    teams = sorted(set(df[df.season == season].home))
    recs = []
    for rd in played["round"].unique():
        games = played[played["round"] == rd]
        cutoff = games.date.min()
        (r, ha, base, rho), _ = model_for(frames, code, season, prev, cutoff, teams)
        for _, g in games.iterrows():
            if g.home not in r.index or g.away not in r.index:
                continue
            lh, la, grid = predict(g.home, g.away, r, ha, base, rho)
            ph, pdr, pa = outcome_probs(grid)
            gg = np.arange(grid.shape[0])
            over = float(grid[np.add.outer(gg, gg) > 2].sum())
            btts = float(grid[1:, 1:].sum())
            res = "H" if g.hg > g.ag else "D" if g.hg == g.ag else "A"
            pick = ["H", "D", "A"][int(np.argmax([ph, pdr, pa]))]
            recs.append(dict(round=rd, date=str(g.date.date()), home=g.home, away=g.away,
                             hg=int(g.hg), ag=int(g.ag), ph=float(ph), pd=float(pdr), pa=float(pa),
                             over=over, btts=btts, xh=float(lh), xa=float(la), pick=pick, res=res,
                             hit=pick == res, over_hit=(over >= 0.5) == (g.hg + g.ag > 2.5),
                             btts_hit=(btts >= 0.5) == (g.hg > 0 and g.ag > 0)))
    return recs


def score_records(recs):
    if not recs:
        return None
    base_p = {"H": 0.45, "D": 0.25, "A": 0.30}
    n = len(recs)
    ll = np.mean([-np.log({"H": r["ph"], "D": r["pd"], "A": r["pa"]}[r["res"]]) for r in recs])
    llb = np.mean([-np.log(base_p[r["res"]]) for r in recs])
    br = np.mean([sum(({"H": r["ph"], "D": r["pd"], "A": r["pa"]}[k] - (r["res"] == k)) ** 2 for k in "HDA") for r in recs])
    brb = np.mean([sum((base_p[k] - (r["res"] == k)) ** 2 for k in "HDA") for r in recs])
    return dict(games=n, accuracy=float(np.mean([r["hit"] for r in recs])),
                over_accuracy=float(np.mean([r["over_hit"] for r in recs])),
                btts_accuracy=float(np.mean([r["btts_hit"] for r in recs])),
                logloss_model=float(ll), logloss_baseline=float(llb),
                brier_model=float(br), brier_baseline=float(brb))


# ---------------- season projections ----------------

SIMS = 10000


def project_season(df, r, ha, base, zones, table, seed=7):
    rng = np.random.default_rng(seed)
    teams = [row["team"] for row in table]
    idx = {t: i for i, t in enumerate(teams)}
    T = len(teams)
    pts = np.tile(np.array([row["pts"] for row in table], float)[:, None], (1, SIMS))
    gd = np.tile(np.array([row["gd"] for row in table], float)[:, None], (1, SIMS))
    gf = np.tile(np.array([row["gf"] for row in table], float)[:, None], (1, SIMS))
    rest = df[(df.season == CURRENT) & df.hg.isna()]
    rest = rest[rest.home.isin(idx) & rest.away.isin(idx) & rest.home.isin(r.index) & rest.away.isin(r.index)]
    for _, g in rest.iterrows():
        lh = np.exp(base + r.attack[g.home] - r.defense[g.away] + ha / 2)
        la = np.exp(base + r.attack[g.away] - r.defense[g.home] - ha / 2)
        x, y = rng.poisson(lh, SIMS), rng.poisson(la, SIMS)
        i, j = idx[g.home], idx[g.away]
        pts[i] += np.where(x > y, 3, np.where(x == y, 1, 0))
        pts[j] += np.where(y > x, 3, np.where(x == y, 1, 0))
        gd[i] += x - y; gd[j] += y - x
        gf[i] += x; gf[j] += y
    score = pts * 1e6 + gd * 1e3 + gf + rng.random((T, SIMS))
    order = np.argsort(-score, axis=0)
    pos = np.empty_like(order)
    pos[order, np.arange(SIMS)] = np.arange(1, T + 1)[:, None]
    out = []
    for t in teams:
        i = idx[t]
        row = dict(team=t, pts_now=int(table[i]["pts"]), played=int(table[i]["p"]),
                   pts_proj=round(float(pts[i].mean()), 1), pos_avg=round(float(pos[i].mean()), 1),
                   zones={})
        for z in zones:
            conv = lambda v: v if v > 0 else T + v + 1
            lo, hi = conv(z["from"]), conv(z["to"])
            row["zones"][z["key"]] = round(float(((pos[i] >= lo) & (pos[i] <= hi)).mean()), 4)
        out.append(row)
    out.sort(key=lambda x: (x["pos_avg"], -x["pts_proj"]))
    return dict(sims=SIMS, remaining=int(len(rest)), zones=zones, teams=out)


# ---------------- bookmaker odds (football-data.co.uk) ----------------

ODDS_URL = "https://www.football-data.co.uk/fixtures.csv"
FD_ALIAS = {
    "Brighton & Hove Albion": "Brighton", "Coventry City": "Coventry", "Hull City": "Hull", "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds", "Manchester City": "Man City", "Manchester United": "Man United",
    "Newcastle United": "Newcastle", "Nottingham Forest": "Nott'm Forest", "Tottenham Hotspur": "Tottenham",
    "Athletic Club": "Ath Bilbao", "CA Osasuna": "Osasuna", "Club Atlético de Madrid": "Ath Madrid",
    "Deportivo Alavés": "Alaves", "Elche CF": "Elche", "Getafe CF": "Getafe", "Levante UD": "Levante",
    "Málaga CF": "Malaga", "FC Barcelona": "Barcelona", "RC Celta de Vigo": "Celta", "RC Deportivo La Coruña": "La Coruna",
    "RCD Espanyol de Barcelona": "Espanol", "Rayo Vallecano de Madrid": "Vallecano", "Real Betis Balompié": "Betis",
    "Real Madrid CF": "Real Madrid", "Real Racing Club de Santander": "Santander", "Real Sociedad de Fútbol": "Sociedad",
    "Valencia CF": "Valencia", "Villarreal CF": "Villarreal",
    "1. FC Köln": "FC Koln", "1. FC Union Berlin": "Union Berlin", "1. FSV Mainz 05": "Mainz",
    "Bayer 04 Leverkusen": "Leverkusen", "Borussia Dortmund": "Dortmund", "Borussia Mönchengladbach": "M'gladbach",
    "Eintracht Frankfurt": "Ein Frankfurt", "FC Augsburg": "Augsburg", "FC Bayern München": "Bayern Munich",
    "FC Schalke 04": "Schalke 04", "Hamburger SV": "Hamburg", "SC Freiburg": "Freiburg", "SC Paderborn 07": "Paderborn",
    "SV 07 Elversberg": "Elversberg", "SV Werder Bremen": "Werder Bremen", "TSG 1899 Hoffenheim": "Hoffenheim",
    "VfB Stuttgart": "Stuttgart",
    "AC Milan": "Milan", "AC Monza": "Monza", "ACF Fiorentina": "Fiorentina", "AS Roma": "Roma", "Atalanta BC": "Atalanta",
    "Bologna FC 1909": "Bologna", "Cagliari Calcio": "Cagliari", "Como 1907": "Como", "FC Internazionale Milano": "Inter",
    "Frosinone Calcio": "Frosinone", "Genoa CFC": "Genoa", "Parma Calcio 1913": "Parma", "SS Lazio": "Lazio",
    "SSC Napoli": "Napoli", "US Lecce": "Lecce", "US Sassuolo Calcio": "Sassuolo", "Udinese Calcio": "Udinese",
    "AJ Auxerre": "Auxerre", "AS Monaco": "Monaco", "Angers SCO": "Angers", "ES Troyes AC": "Troyes", "FC Lorient": "Lorient",
    "Le Havre AC": "Le Havre", "Lille OSC": "Lille", "OGC Nice": "Nice", "Olympique Lyonnais": "Lyon",
    "Olympique de Marseille": "Marseille", "Paris": "Paris FC", "Paris Saint-Germain": "Paris SG",
    "RC Strasbourg Alsace": "Strasbourg", "Racing Club de Lens": "Lens", "Stade Brestois 29": "Brest",
    "Stade Rennais FC 1901": "Rennes",
    "Birmingham City": "Birmingham", "Blackburn Rovers": "Blackburn", "Bolton Wanderers": "Bolton", "Cardiff City": "Cardiff",
    "Charlton Athletic": "Charlton", "Derby County": "Derby", "Lincoln City": "Lincoln", "Norwich City": "Norwich",
    "Preston North End": "Preston", "Queens Park Rangers": "QPR", "Stoke City": "Stoke", "Swansea City": "Swansea",
    "West Bromwich Albion": "West Brom", "West Ham United": "West Ham", "Wolverhampton Wanderers": "Wolves",
    "ADO Den Haag": "Den Haag", "AZ": "AZ Alkmaar", "FC Groningen": "Groningen", "FC Twente '65": "Twente",
    "FC Utrecht": "Utrecht", "Feyenoord Rotterdam": "Feyenoord", "Fortuna Sittard": "For Sittard", "NEC": "Nijmegen",
    "PEC Zwolle": "Zwolle", "PSV": "PSV Eindhoven", "SBV Excelsior": "Excelsior", "SC Cambuur-Leeuwarden": "Cambuur",
    "SC Heerenveen": "Heerenveen", "Telstar 1963": "Telstar", "Willem II Tilburg": "Willem II",
    "Académico de Viseu": "Academico Viseu", "CD Nacional": "Nacional", "CD Santa Clara": "Santa Clara",
    "CF Estrela da Amadora": "Estrela", "CS Marítimo": "Maritimo", "Casa Pia AC": "Casa Pia", "FC Alverca": "Alverca",
    "FC Arouca": "Arouca", "FC Famalicão": "Famalicao", "FC Porto": "Porto", "GD Estoril Praia": "Estoril",
    "Sport Lisboa e Benfica": "Benfica", "Sporting Clube de Braga": "Sp Braga", "Sporting Clube de Portugal": "Sp Lisbon",
    "Vitória Guimarães": "Guimaraes",
}


def _norm(s):
    import unicodedata
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode().lower()
    return "".join(ch for ch in s if ch.isalnum())


def fetch_odds():
    """returns list of fixture rows, or None if the odds site can't be reached."""
    import csv
    import io
    import requests
    try:
        r = requests.get(ODDS_URL, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print("odds unavailable:", type(e).__name__)
        return None
    raw = r.content
    for enc in ("utf-8-sig", "latin-1"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    return list(csv.DictReader(io.StringIO(text)))


def _num(x):
    try:
        v = float(x)
        return v if v > 1.0 else None
    except (TypeError, ValueError):
        return None


def attach_odds(rows, fd_code, matches):
    """adds an 'odds' block to every match we can find in the bookmaker file."""
    import difflib
    if rows is None:
        return None
    mine = [r for r in rows if (r.get("Div") or "").strip() == fd_code]
    their_teams = sorted({r["HomeTeam"].strip() for r in mine} | {r["AwayTeam"].strip() for r in mine})
    lookup = {_norm(t): t for t in their_teams}

    def to_fd(name):
        alias = FD_ALIAS.get(name, name)
        if _norm(alias) in lookup:
            return lookup[_norm(alias)]
        best = difflib.get_close_matches(_norm(alias), list(lookup), n=1, cutoff=0.6)
        return lookup[best[0]] if best else None

    index = {(r["HomeTeam"].strip(), r["AwayTeam"].strip()): r for r in mine}
    found = 0
    for m in matches:
        key = (to_fd(m["home"]), to_fd(m["away"]))
        row = index.get(key)
        if not row:
            continue
        pick = lambda *keys: next((_num(row.get(k)) for k in keys if _num(row.get(k))), None)
        o = dict(
            H=pick("AvgH", "B365H", "PSH"), D=pick("AvgD", "B365D", "PSD"), A=pick("AvgA", "B365A", "PSA"),
            O25=pick("Avg>2.5", "B365>2.5", "P>2.5"), U25=pick("Avg<2.5", "B365<2.5", "P<2.5"),
            bestH=pick("MaxH"), bestD=pick("MaxD"), bestA=pick("MaxA"),
        )
        if not (o["H"] and o["D"] and o["A"]):
            continue
        p = m["pred"]
        inv = np.array([1 / o["H"], 1 / o["D"], 1 / o["A"]])
        mk = inv / inv.sum()
        lines = []
        for k, mp, mkp, odd in (("H", p["home_win"], mk[0], o["H"]), ("D", p["draw"], mk[1], o["D"]),
                                ("A", p["away_win"], mk[2], o["A"])):
            lines.append(dict(key=k, model=mp, market=float(mkp), odds=odd, ev=mp * odd - 1))
        if o["O25"] and o["U25"]:
            iv = np.array([1 / o["O25"], 1 / o["U25"]])
            mko = iv / iv.sum()
            lines.append(dict(key="O2.5", model=p["over25"], market=float(mko[0]), odds=o["O25"], ev=p["over25"] * o["O25"] - 1))
            lines.append(dict(key="U2.5", model=1 - p["over25"], market=float(mko[1]), odds=o["U25"], ev=(1 - p["over25"]) * o["U25"] - 1))
        m["odds"] = dict(source="football-data.co.uk average prices", lines=lines,
                         margin=float(inv.sum() - 1))
        found += 1
    return found


# ---------------- stats helpers ----------------

def team_games(played, team):
    hm = played[played.home == team].assign(venue="H", gf=lambda d: d.hg, ga=lambda d: d.ag,
                                            opp=lambda d: d.away, gfh=lambda d: d.hht, gah=lambda d: d.aht)
    aw = played[played.away == team].assign(venue="A", gf=lambda d: d.ag, ga=lambda d: d.hg,
                                            opp=lambda d: d.home, gfh=lambda d: d.aht, gah=lambda d: d.hht)
    return pd.concat([hm, aw]).sort_values("date")


def result(gf, ga):
    return "W" if gf > ga else "D" if gf == ga else "L"


def summarize(g):
    if len(g) == 0:
        return None
    res = [result(a, b) for a, b in zip(g.gf, g.ga)]
    pts = sum(3 if x == "W" else 1 if x == "D" else 0 for x in res)
    tot = g.gf + g.ga
    return dict(
        played=len(g), w=res.count("W"), d=res.count("D"), l=res.count("L"),
        ppg=round(pts / len(g), 2),
        gf=round(g.gf.mean(), 2), ga=round(g.ga.mean(), 2),
        clean_sheets=round((g.ga == 0).mean(), 3),
        failed_to_score=round((g.gf == 0).mean(), 3),
        over25=round((tot > 2.5).mean(), 3),
        btts=round(((g.gf > 0) & (g.ga > 0)).mean(), 3),
    )


def form_list(g, k=5):
    out = []
    for _, x in g.tail(k).iloc[::-1].iterrows():
        out.append(dict(date=str(x.date.date()), opp=x.opp, venue=x.venue,
                        gf=int(x.gf), ga=int(x.ga), res=result(x.gf, x.ga)))
    return out


def league_table(played_cur):
    teams = sorted(set(played_cur.home) | set(played_cur.away))
    rows = []
    for t in teams:
        s = summarize(team_games(played_cur, t))
        gtot = team_games(played_cur, t)
        rows.append(dict(team=t, p=s["played"], w=s["w"], d=s["d"], l=s["l"],
                         gf=int(gtot.gf.sum()), ga=int(gtot.ga.sum()),
                         gd=int(gtot.gf.sum() - gtot.ga.sum()),
                         pts=s["w"] * 3 + s["d"]))
    rows.sort(key=lambda r: (-r["pts"], -r["gd"], -r["gf"], r["team"]))
    for i, r in enumerate(rows, 1):
        r["pos"] = i
    return rows


def pct(x):
    return f"{round(x * 100)}%"


def form_tone(ppg):
    return "good" if ppg >= 1.8 else "bad" if ppg <= 1.0 else None


def insights(home, away, p, hn, an, h2h_sum, h2h_games, hvenue, avenue, hform, aform, r):
    out = []
    ph, pd_, pa = p["home_win"], p["draw"], p["away_win"]
    fav = home if ph > pa else away
    fav_p = max(ph, pa)

    # why the model leans this way
    adiff = r.attack[home] - r.attack[away]
    ddiff = r.defense[home] - r.defense[away]
    lean = []
    at = (home if adiff > 0 else away) if abs(adiff) > 0.1 else None
    dt = (home if ddiff > 0 else away) if abs(ddiff) > 0.1 else None
    if at and at == dt:
        lean.append(f"{at} rate better in both attack and defense")
    else:
        if at:
            lean.append(f"{at} have the stronger attack")
        if dt:
            lean.append(f"{dt} have the stronger defense")
    lean.append(f"{home} get the home boost")
    if abs(ph - pa) < 0.08:
        out.append(dict(kind="why", text=f"Basically a coin flip ({pct(ph)} vs {pct(pa)}): " + ", ".join(lean) + "."))
    else:
        out.append(dict(kind="why", text=f"Model leans {fav} ({pct(fav_p)}): " + ", ".join(lean) + "."))

    if pd_ >= 0.27:
        out.append(dict(kind="why", text=f"Draw is live at {pct(pd_)}. These sides rate closely and neither projects many goals."))

    # venue form
    if hvenue and hvenue["played"] >= 3:
        out.append(dict(kind="trend", tone=form_tone(hvenue["ppg"]), text=f"{home} at home since last season: {hvenue['w']}W {hvenue['d']}D {hvenue['l']}L, "
                                           f"{hvenue['gf']} scored / {hvenue['ga']} conceded per game."))
    if avenue and avenue["played"] >= 3:
        out.append(dict(kind="trend", tone=form_tone(avenue["ppg"]), text=f"{away} away since last season: {avenue['w']}W {avenue['d']}D {avenue['l']}L, "
                                           f"{avenue['gf']} scored / {avenue['ga']} conceded per game."))

    # streaks
    for team, f in ((home, hform), (away, aform)):
        res = [x["res"] for x in f]
        if len(res) >= 3 and all(x == res[0] for x in res[:3]):
            word = {"W": "won", "D": "drawn", "L": "lost"}[res[0]]
            k = 0
            for x in res:
                if x == res[0]:
                    k += 1
                else:
                    break
            out.append(dict(kind="trend", tone={"W": "good", "D": None, "L": "bad"}[res[0]], text=f"{team} have {word} their last {k} league games."))
        nogoal = sum(1 for x in f if x["gf"] == 0)
        if nogoal >= 3:
            out.append(dict(kind="watch", tone="bad", text=f"{team} failed to score in {nogoal} of their last {len(f)}."))
        cs = sum(1 for x in f if x["ga"] == 0)
        if cs >= 3:
            out.append(dict(kind="watch", tone="good", text=f"{team} kept {cs} clean sheets in their last {len(f)}."))

    # goals angle
    if p["over25"] >= 0.6:
        out.append(dict(kind="watch", text=f"Goals expected: model has over 2.5 at {pct(p['over25'])} and BTTS at {pct(p['btts'])}."))
    elif p["over25"] <= 0.42:
        out.append(dict(kind="watch", text=f"Tight one projected: under 2.5 at {pct(1 - p['over25'])}."))

    # h2h
    if h2h_sum and h2h_sum["meetings"] >= 3:
        t = h2h_sum
        out.append(dict(kind="h2h", text=f"H2H: {home} {t['home_team_wins']}W, {t['draws']}D, {away} {t['away_team_wins']}W "
                                         f"in {t['meetings']} league meetings, {t['avg_goals']} goals per game."))
        if t["over25"] >= 0.65:
            out.append(dict(kind="h2h", text=f"This fixture tends to open up: {pct(t['over25'])} of meetings went over 2.5."))
        if t["over25"] <= 0.35:
            out.append(dict(kind="h2h", text=f"This fixture tends to be cagey: only {pct(t['over25'])} of meetings went over 2.5."))
        recent = h2h_games[:5]
        if len(recent) >= 3:
            hw = sum(1 for g in recent if g["winner"] == home)
            aw = sum(1 for g in recent if g["winner"] == away)
            if hw >= 4 or aw >= 4:
                dom = home if hw > aw else away
                out.append(dict(kind="h2h", text=f"{dom} have won {max(hw, aw)} of the last {len(recent)} meetings."))
    elif not h2h_sum or h2h_sum["meetings"] == 0:
        out.append(dict(kind="h2h", text="No league meetings in the data, so h2h tells us nothing here."))

    # small sample warning
    for team, n in ((home, hn), (away, an)):
        if n < 15:
            out.append(dict(kind="caution", tone="warn", text=f"{team} have limited recent data in this league, so their ratings are shakier than most."))
    return out


# ---------------- main ----------------

def build_league(code, name, country, fd_code, zones, frames, h2h_pool, today, odds_rows):
    df = frames[code]
    played = df.dropna(subset=["hg"])
    cur_teams = sorted(set(df[df.season == CURRENT].home))
    (r, ha, base, rho), n_games = model_for(frames, code, CURRENT, LAST, today + pd.Timedelta(days=1), cur_teams)
    model_window = played[played.season.isin([LAST, CURRENT])]

    upcoming = df[df.hg.isna() & (df.season == CURRENT) & (df.date >= today - pd.Timedelta(days=1))]
    upcoming = upcoming[upcoming.home.isin(r.index) & upcoming.away.isin(r.index)]
    if upcoming.empty:
        rd, games = None, upcoming
    else:
        rd = upcoming.sort_values("date")["round"].iloc[0]
        games = upcoming[upcoming["round"] == rd].sort_values(["date", "time"])

    played_cur = played[played.season == CURRENT]
    recent2 = model_window  # last season + this season for venue splits

    matches = []
    for _, g in games.iterrows():
        home, away = g.home, g.away
        lh, la, grid = predict(home, away, r, ha, base, rho)
        ph, pdraw, pa = outcome_probs(grid)
        gg = np.arange(grid.shape[0])
        over25 = float(grid[np.add.outer(gg, gg) > 2].sum())
        btts = float(grid[1:, 1:].sum())
        flat = [(int(i), int(j), float(grid[i, j])) for i in gg for j in gg]
        flat.sort(key=lambda x: -x[2])

        hp = h2h_pool
        hh = hp[((hp.home == home) & (hp.away == away)) | ((hp.home == away) & (hp.away == home))]
        hh = hh.sort_values("date", ascending=False)
        h2h_games = []
        for _, x in hh.iterrows():
            winner = x.home if x.hg > x.ag else x.away if x.ag > x.hg else None
            h2h_games.append(dict(date=str(x.date.date()), season=x.season, home=x.home, away=x.away,
                                  hg=int(x.hg), ag=int(x.ag), winner=winner))
        if len(hh):
            tot = hh.hg + hh.ag
            h2h_sum = dict(
                meetings=len(hh),
                home_team_wins=sum(1 for x in h2h_games if x["winner"] == home),
                away_team_wins=sum(1 for x in h2h_games if x["winner"] == away),
                draws=sum(1 for x in h2h_games if x["winner"] is None),
                avg_goals=round(float(tot.mean()), 2),
                over25=round(float((tot > 2.5).mean()), 3),
                btts=round(float(((hh.hg > 0) & (hh.ag > 0)).mean()), 3),
                at_this_venue=sum(1 for x in h2h_games if x["home"] == home),
                home_wins_at_venue=sum(1 for x in h2h_games if x["home"] == home and x["winner"] == home),
            )
        else:
            h2h_sum = dict(meetings=0)

        hg_all = team_games(recent2, home)
        ag_all = team_games(recent2, away)
        h_season = summarize(team_games(played_cur, home))
        a_season = summarize(team_games(played_cur, away))
        hvenue = summarize(hg_all[hg_all.venue == "H"])
        avenue = summarize(ag_all[ag_all.venue == "A"])
        hform = form_list(team_games(played, home))
        aform = form_list(team_games(played, away))

        pred = dict(home_win=float(ph), draw=float(pdraw), away_win=float(pa),
                    over25=over25, btts=btts, xg_home=float(lh), xg_away=float(la))
        g7 = grid[:GRID, :GRID]
        calc = dict(
            base_goals=float(np.exp(base)), home_adv=ha, home_adv_mult=float(np.exp(ha)), rho=rho,
            home_att=float(r.attack[home]), home_def=float(r.defense[home]),
            away_att=float(r.attack[away]), away_def=float(r.defense[away]),
            home_goal_dist=[float(x) for x in poisson.pmf(np.arange(GRID), lh)],
            away_goal_dist=[float(x) for x in poisson.pmf(np.arange(GRID), la)],
        )
        matches.append(dict(
            id=(code + "-" + home + "-" + away).replace(" ", "_").replace("&", "and").replace(".", "").replace("'", ""),
            date=str(g.date.date()), time=g.time, home=home, away=away,
            pred=pred, calc=calc,
            grid=[[round(float(g7[i, j]), 6) for j in range(GRID)] for i in range(GRID)],
            mgrid=[[round(float(grid[i, j]), 5) for j in range(10)] for i in range(10)],  # 0-9 goals, for markets
            top_scores=[dict(h=a, a=b, p=c) for a, b, c in flat[:5]],
            h2h=dict(summary=h2h_sum, games=h2h_games[:10]),
            teams=dict(
                home=dict(season=h_season, venue=hvenue, form=hform),
                away=dict(season=a_season, venue=avenue, form=aform),
            ),
            insights=insights(home, away, dict(home_win=ph, draw=pdraw, away_win=pa, over25=over25, btts=btts),
                              len(hg_all), len(ag_all), h2h_sum, h2h_games, hvenue, avenue, hform, aform, r),
        ))

    teams_rated = [t for t in cur_teams if t in r.index]
    rk = r.loc[teams_rated].assign(overall=lambda d: d.attack + d.defense).sort_values("overall", ascending=False)
    ratings = [dict(team=t, attack=float(x.attack), defense=float(x.defense), overall=float(x.overall),
                    xg_for_vs_avg=float(np.exp(base + x.attack)), xg_against_vs_avg=float(np.exp(base - x.defense)))
               for t, x in rk.iterrows()]

    table = league_table(played_cur)
    zl = zone_list(zones)
    projections = project_season(df, r, ha, base, zl, table)

    recs = walk_forward(frames, code, CURRENT, LAST)
    played_rounds = sorted({x["round"] for x in recs}, key=lambda rr: min(x["date"] for x in recs if x["round"] == rr))
    receipts = dict(season=score_records(recs),
                    rounds=[dict(round=rr, games=[x for x in recs if x["round"] == rr]) for rr in played_rounds[::-1][:3]])
    bt = score_records(walk_forward(frames, code, LAST, PREV)) if (df.season == PREV).any() else None
    if bt:
        bt["season"] = LAST

    odds_found = attach_odds(odds_rows, fd_code, matches)
    first = sorted(h2h_pool.season.unique())[0]

    return dict(
        code=code, name=name, country=country,
        results_through=str(played.date.max().date()),
        matchday=rd, trained_on=int(n_games),
        seasons_used=[LAST, CURRENT], h2h_from=first, joint=code in JOINT,
        home_adv_mult=float(np.exp(ha)), base_goals=float(np.exp(base)), rho=rho,
        matches=matches, ratings=ratings, table=table, zones=zl,
        projections=projections, receipts=receipts, backtest=bt,
        odds_status=("unavailable" if odds_found is None else f"{odds_found} of {len(matches)} games"),
    )


def main():
    today = pd.Timestamp(date.today())
    frames = {}
    for code, name, *_ in LEAGUES:
        print("downloading", name)
        download(code)
        frames[code] = load(code)
    odds_rows = fetch_odds()

    leagues = []
    for code, name, country, fd_code, zones in LEAGUES:
        same_country = [frames[c] for c, _, k, *_ in LEAGUES if k == country]
        h2h_pool = pd.concat(same_country).dropna(subset=["hg"]).drop_duplicates(["date", "home", "away"])
        lg = build_league(code, name, country, fd_code, zones, frames, h2h_pool, today, odds_rows)
        bt, rc = lg["backtest"], lg["receipts"]["season"]
        print(f"{name}: {lg['matchday']} | {len(lg['matches'])} games | rho {lg['rho']:.3f} | odds {lg['odds_status']}"
              + (f" | last season acc {bt['accuracy']:.1%} ll {bt['logloss_model']:.3f} vs {bt['logloss_baseline']:.3f}" if bt else "")
              + (f" | this season {rc['accuracy']:.1%} of {rc['games']}" if rc else ""))
        leagues.append(lg)

    out = dict(generated=str(today.date()),
               settings=dict(half_life_days=HALF_LIFE_DAYS, penalty=PENALTY, sims=SIMS),
               odds_available=odds_rows is not None,
               leagues=leagues)
    json.dump(out, open("site_data.json", "w"), separators=(",", ":"))
    print("bytes:", os.path.getsize("site_data.json"))


if __name__ == "__main__":
    main()
