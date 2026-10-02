# Ball Knowledge

Match predictions for 8 European leagues: Premier League, La Liga, Bundesliga, Serie A, Ligue 1, Championship, Eredivisie and Primeira Liga.

Every prediction shows the full math behind it: team attack and defense ratings, expected goals, a Poisson goals model, and every possible scoreline added up into win / draw / loss odds. Plus head to head, form, home and away splits, and how each league's model would have done last season.

**How it updates:** a GitHub Action runs every morning, pulls the latest results, refits the model and rebuilds `index.html`.

**Data:** [openfootball/football.json](https://github.com/openfootball/football.json)

Not betting advice. The model is simple and bookmakers are sharper.
