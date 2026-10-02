"""
builds site_data.json: predictions + the full math + h2h + form + insights
for the next matchday in 8 european leagues.

data: openfootball/football.json on github (results back to 2010-11)
run:  python build_data.py   (downloads the data itself)
"""

import glob
import json
import os
from datetime import date

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

HALF_LIFE_DAYS = 180
PENALTY = 0.5
GRID = 7  # show 0..6 goals in the score grid
CURRENT = "2026-27"
LAST = "2025-26"
PREV = "2024-25"
FIRST_YEAR = 2010
URL = "https://raw.githubusercontent.com/openfootball/football.json/master/{season}/{code}.json"

LEAGUES = [
    # code, name, country
    ("en.1", "Premier League", "England"),
    ("es.1", "La Liga", "Spain"),
    ("de.1", "Bundesliga", "Germany"),
    ("it.1", "Serie A", "Italy"),
    ("fr.1", "Ligue 1", "France"),
    ("en.2", "Championship", "England"),
    ("nl.1", "Eredivisie", "Netherlands"),
    ("pt.1", "Primeira Liga", "Portugal"),
]


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

def fit(played, today):
    teams = sorted(set(played.home) | set(played.away))
    idx = {t: i for i, t in enumerate(teams)}
    n = len(teams)
    h, a = played.home.map(idx).values, played.away.map(idx).values
    hg, ag = played.hg.values, played.ag.values
    w = 0.5 ** ((today - played.date).dt.days.values / HALF_LIFE_DAYS)

    def nll(p):
        att, dfn, ha = p[:n], p[n:2 * n], p[-1]
        lh = np.exp(att[h] - dfn[a] + ha)
        la = np.exp(att[a] - dfn[h])
        ll = w * (poisson.logpmf(hg, lh) + poisson.logpmf(ag, la))
        return -ll.sum() + PENALTY * (np.sum(att ** 2) + np.sum(dfn ** 2))

    x0 = np.zeros(2 * n + 1)
    x0[-1] = 0.2
    p = minimize(nll, x0, method="L-BFGS-B").x
    att = p[:n] - p[:n].mean()
    dfn = p[n:2 * n] - p[n:2 * n].mean()
    base = float(np.log(np.average(np.r_[hg, ag], weights=np.r_[w, w])))
    r = pd.DataFrame({"attack": att, "defense": dfn}, index=teams)
    return r, float(p[-1]), base


def predict(home, away, r, ha, base, grid_n=11):
    lh = np.exp(base + r.attack[home] - r.defense[away] + ha / 2)
    la = np.exp(base + r.attack[away] - r.defense[home] - ha / 2)
    g = np.arange(grid_n)
    grid = np.outer(poisson.pmf(g, lh), poisson.pmf(g, la))
    return lh, la, grid


def outcome_probs(grid):
    return np.tril(grid, -1).sum(), np.trace(grid), np.triu(grid, 1).sum()


# ---------------- backtest (last season, walk forward) ----------------

def backtest(df):
    played = df.dropna(subset=["hg"])
    test = played[played.season == LAST]
    train_pool = played[played.season.isin([PREV, LAST])]
    rounds = test["round"].unique()
    ll_m, ll_b, hit, brier_m, brier_b, n = 0, 0, 0, 0, 0, 0
    base_p = np.array([0.45, 0.25, 0.30])
    for rd in rounds:
        games = test[test["round"] == rd]
        cutoff = games.date.min()
        tr = train_pool[train_pool.date < cutoff]
        r, ha, base = fit(tr, cutoff)
        for _, g in games.iterrows():
            if g.home not in r.index or g.away not in r.index:
                continue
            _, _, grid = predict(g.home, g.away, r, ha, base)
            p = np.array(outcome_probs(grid))
            p = p / p.sum()
            res = 0 if g.hg > g.ag else 1 if g.hg == g.ag else 2
            y = np.eye(3)[res]
            ll_m += -np.log(p[res]); ll_b += -np.log(base_p[res])
            brier_m += ((p - y) ** 2).sum(); brier_b += ((base_p - y) ** 2).sum()
            hit += int(np.argmax(p) == res); n += 1
    return dict(
        season=LAST, games=n, accuracy=hit / n,
        logloss_model=ll_m / n, logloss_baseline=ll_b / n,
        brier_model=brier_m / n, brier_baseline=brier_b / n,
    )


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

def build_league(code, name, country, df, h2h_pool, today):
    played = df.dropna(subset=["hg"])
    model_window = played[played.season.isin([LAST, CURRENT])]
    r, ha, base = fit(model_window, today)

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
        lh, la, grid = predict(home, away, r, ha, base)
        ph, pdraw, pa = outcome_probs(grid)
        gg = np.arange(grid.shape[0])
        over25 = float(sum(grid[i, j] for i in gg for j in gg if i + j > 2))
        btts = float(grid[1:, 1:].sum())
        flat = [(int(i), int(j), float(grid[i, j])) for i in gg for j in gg]
        flat.sort(key=lambda x: -x[2])

        # h2h
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

        calc = dict(
            base=base, base_goals=float(np.exp(base)), home_adv=ha, home_adv_mult=float(np.exp(ha)),
            home_att=float(r.attack[home]), home_def=float(r.defense[home]),
            away_att=float(r.attack[away]), away_def=float(r.defense[away]),
            log_home=float(base + r.attack[home] - r.defense[away] + ha / 2),
            log_away=float(base + r.attack[away] - r.defense[home] - ha / 2),
            home_goal_dist=[float(x) for x in poisson.pmf(np.arange(GRID), lh)],
            away_goal_dist=[float(x) for x in poisson.pmf(np.arange(GRID), la)],
        )

        matches.append(dict(
            id=(code + "-" + home + "-" + away).replace(" ", "_").replace("&", "and").replace(".", ""),
            date=str(g.date.date()), time=g.time, home=home, away=away,
            pred=pred, calc=calc,
            grid=[[float(grid[i, j]) for j in range(GRID)] for i in range(GRID)],
            top_scores=[dict(h=a, a=b, p=c) for a, b, c in flat[:5]],
            h2h=dict(summary=h2h_sum, games=h2h_games[:10]),
            teams=dict(
                home=dict(season=h_season, venue=hvenue, form=hform),
                away=dict(season=a_season, venue=avenue, form=aform),
            ),
            insights=insights(home, away, dict(home_win=ph, draw=pdraw, away_win=pa, over25=over25, btts=btts),
                              len(hg_all), len(ag_all), h2h_sum, h2h_games, hvenue, avenue, hform, aform, r),
        ))

    cur_teams = sorted(set(df[df.season == CURRENT].home) & set(r.index))
    rk = r.loc[cur_teams].assign(overall=lambda d: d.attack + d.defense).sort_values("overall", ascending=False)
    ratings = [dict(team=t, attack=float(x.attack), defense=float(x.defense), overall=float(x.overall),
                    xg_for_vs_avg=float(np.exp(base + x.attack)), xg_against_vs_avg=float(np.exp(base - x.defense)))
               for t, x in rk.iterrows()]

    bt = backtest(df) if (df.season == PREV).any() else None
    first = sorted(h2h_pool.season.unique())[0]

    return dict(
        code=code, name=name, country=country,
        results_through=str(played.date.max().date()),
        matchday=rd, trained_on=int(len(model_window)),
        seasons_used=[LAST, CURRENT], h2h_from=first,
        home_adv_mult=float(np.exp(ha)), base_goals=float(np.exp(base)),
        matches=matches, ratings=ratings, table=league_table(played_cur), backtest=bt,
    )


def main():
    today = pd.Timestamp(date.today())
    frames = {}
    for code, name, country in LEAGUES:
        print("downloading", name)
        download(code)
        frames[code] = load(code)

    leagues = []
    for code, name, country in LEAGUES:
        same_country = [frames[c] for c, _, k in LEAGUES if k == country]
        h2h_pool = pd.concat(same_country).dropna(subset=["hg"]).drop_duplicates(["date", "home", "away"])
        lg = build_league(code, name, country, frames[code], h2h_pool, today)
        bt = lg["backtest"]
        print(f"{name}: {lg['matchday']} | {len(lg['matches'])} games"
              + (f" | backtest acc {bt['accuracy']:.1%} ll {bt['logloss_model']:.3f} vs {bt['logloss_baseline']:.3f}" if bt else ""))
        leagues.append(lg)

    out = dict(generated=str(today.date()),
               settings=dict(half_life_days=HALF_LIFE_DAYS, penalty=PENALTY),
               leagues=leagues)
    json.dump(out, open("site_data.json", "w"), separators=(",", ":"))
    print("bytes:", os.path.getsize("site_data.json"))


if __name__ == "__main__":
    main()
