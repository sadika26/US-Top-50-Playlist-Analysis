import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA = Path("Atlantic_United_States.csv")

def load_data():
    if not DATA.exists():
        raise FileNotFoundError(f"Put {DATA.name} in the same folder as this script.")
    df = pd.read_csv(DATA)
    required = ["date","position","song","artist","popularity","duration_ms","album_type","total_tracks","is_explicit"]
    missing = [c for c in required if c not in df.columns]
    if missing: raise ValueError(f"Missing dataset columns: {missing}")
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y", errors="coerce")
    for c in ["position","popularity","duration_ms","total_tracks"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["song"] = df["song"].astype("string").str.strip()
    df["artist"] = df["artist"].astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    df["album_type"] = df["album_type"].astype("string").str.strip().str.lower()
    df["is_explicit"] = df["is_explicit"].astype("string").str.lower().str.strip().map(
        {"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False})
    before = len(df)
    exact_duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates()
    invalid_ranks = int((~df["position"].between(1,50)).fillna(True).sum())
    df = df.dropna(subset=["date","position","song","artist","popularity"])
    df = df[df["position"].between(1,50)].copy()
    dup_mask = df.duplicated(["date","song","artist"], keep=False)
    duplicate_groups = int(df.loc[dup_mask, ["date","song","artist"]].drop_duplicates().shape[0])
    df = df.sort_values(["date","song","artist","position"]).drop_duplicates(
        ["date","song","artist"], keep="first")
    df["duration_minutes"] = df["duration_ms"] / 60000
    audit = {"Source rows":before, "Exact duplicate rows removed":exact_duplicates,
             "Invalid rank rows detected":invalid_ranks,
             "Duplicate date-song-artist groups resolved":duplicate_groups,
             "Clean rows":len(df)}
    return df.sort_values(["song","artist","date"]).reset_index(drop=True), audit

def main():
    df, audit = load_data()
    g = df.groupby(["song","artist"], dropna=False)
    df["days_on_chart"] = g["date"].transform("nunique")
    df["average_rank"] = g["position"].transform("mean")
    df["best_rank_achieved"] = g["position"].transform("min")
    df["rank_volatility_index"] = g["position"].transform("std").fillna(0)
    df["popularity_trend_score"] = df.groupby(["song","artist"], dropna=False)["popularity"].transform(
        lambda s: s.rolling(7, min_periods=1).mean())
    df["rank_change"] = g["position"].diff()
    df["rank_movement"] = df["rank_change"].map(
        lambda x:"First observation" if pd.isna(x) else "Improved" if x<0 else "Declined" if x>0 else "Unchanged")
    df["duration_category"] = pd.cut(df["duration_minutes"], [0,3,4,float("inf")],
        labels=["Short (<3 min)","Medium (3–4 min)","Long (4+ min)"], right=False)
    songs = df.groupby(["song","artist"]).agg(
        days_on_chart=("date","nunique"), average_rank=("position","mean"),
        best_rank_achieved=("position","min"), rank_volatility_index=("position","std"),
        average_popularity=("popularity","mean"), popularity_trend_score=("popularity_trend_score","last"),
        observations=("position","size")).reset_index()
    songs["rank_volatility_index"] = songs["rank_volatility_index"].fillna(0)
    entries = df.groupby("artist").size().sort_values(ascending=False)
    artist_days = df.groupby("artist")["date"].nunique().sort_values(ascending=False)
    dominance = (entries / len(df) * 100).rename("artist_dominance_index_percent")
    bins = pd.cut(df["position"], [0,10,20,30,40,50], labels=["1-10","11-20","21-30","31-40","41-50"])
    daily_sets = df.groupby("date").apply(lambda x:set(zip(x.song,x.artist)), include_groups=False)
    daily=[]; prev=set()
    for date, current in daily_sets.items():
        daily.append({"date":date,"new_song_entries":len(current-prev),"song_exits":len(prev-current)})
        prev=current
    summary = [
        "UNITED STATES TOP 50 PLAYLIST ANALYSIS",
        f"Date range: {df.date.min().date()} to {df.date.max().date()}",
        f"Clean entries: {len(df):,}",
        f"Unique song-artist pairs: {df[['song','artist']].drop_duplicates().shape[0]:,}",
        f"Unique artists: {df.artist.nunique():,}",
        f"Average popularity: {df.popularity.mean():.2f}",
        f"Average duration: {df.duration_minutes.mean():.2f} minutes",
        "", "POPULARITY BAND COUNTS:",
df["popularity"].pipe(lambda s: pd.cut(s, [-1, 61, 71, 81, 91, 101], labels=["0-60", "61-70", "71-80", "81-90", "91-100"]).value_counts().sort_index()).to_string(),
"", "DURATION CATEGORY COUNTS:",
        df["duration_category"].value_counts().sort_index().to_string(),
        f"Popularity-position correlation: {df.popularity.corr(df.position):.3f}",
        f"Explicit content share: {df.is_explicit.eq(True).mean()*100:.2f}%",
        "", "CLEANING AUDIT:", pd.Series(audit).to_string(),
        "", "POPULARITY BY RANK RANGE:", df.assign(rank_range=bins).groupby("rank_range", observed=False).popularity.mean().round(2).to_string(),
        "", "ALBUM TYPE COMPARISON:", df.groupby("album_type").agg(entries=("song","size"), average_popularity=("popularity","mean"), average_rank=("position","mean"), average_duration_minutes=("duration_minutes","mean"), average_total_tracks=("total_tracks","mean")).round(2).to_string(),
        "", "EXPLICIT VS NON-EXPLICIT:", df.groupby("is_explicit").agg(entries=("song","size"), average_popularity=("popularity","mean"), average_rank=("position","mean")).round(2).to_string(),
        "", "TOP SONGS BY LONGEST PRESENCE:", songs.sort_values("days_on_chart", ascending=False).head(10).round(2).to_string(index=False),
        "", "TOP SONGS BY BEST RANK:", songs.sort_values("best_rank_achieved").head(10).round(2).to_string(index=False),
        "", "TOP ARTISTS BY ENTRIES:", entries.head(10).to_string(),
        "", "TOP ARTISTS BY DISTINCT DAYS:", artist_days.head(10).to_string(),
        "", "ARTIST DOMINANCE INDEX (% of all entries):", dominance.sort_values(ascending=False).head(10).round(2).to_string(),
        "", "RANK MOVEMENT COUNTS:", df.rank_movement.value_counts().to_string(),
        "", "FAST RISERS (largest one-step rank improvements):", df.dropna(subset=["rank_change"]).sort_values("rank_change").head(10)[["date","song","artist","rank_change","position"]].to_string(index=False),
        "", "SLOW DECLINERS (largest one-step rank declines):", df.dropna(subset=["rank_change"]).sort_values("rank_change", ascending=False).head(10)[["date","song","artist","rank_change","position"]].to_string(index=False),
        "", "DAILY ENTRY/EXIT COUNTS:", pd.DataFrame(daily).to_string(index=False)
    ]
    summary.extend(["", "POPULARITY BAND COUNTS:", df["popularity"].pipe(lambda s: pd.cut(s, [-1, 61, 71, 81, 91, 101], labels=["0-60", "61-70", "71-80", "81-90", "91-100"]).value_counts().sort_index().to_string()), "", "DURATION CATEGORY COUNTS:", df["duration_category"].value_counts().sort_index().to_string()])
    summary.extend([
    "",
    "AVERAGE POPULARITY BY DURATION CATEGORY:",
    df.groupby("duration_category")["popularity"].mean().sort_index().round(2).to_string()
])
    Path("analysis_summary.txt").write_text("\n".join(summary), encoding="utf-8")
    songs.to_csv("song_kpi_summary.csv", index=False)
    dominance.to_csv("artist_dominance_summary.csv")
    pd.DataFrame(daily).to_csv("daily_entries_exits.csv", index=False)
    monthly = df.groupby(df.date.dt.to_period("M")).popularity.mean()
    fig, ax = plt.subplots(figsize=(11,5)); ax.plot(monthly.index.astype(str), monthly.values, marker="o")
    ax.set(title="Monthly Average Song Popularity", xlabel="Month", ylabel="Average popularity"); ax.tick_params(axis="x", rotation=45); ax.grid(alpha=.25); fig.tight_layout(); fig.savefig("monthly_popularity_trend.png", dpi=200); plt.close(fig)
    df.groupby("artist").popularity.mean().sort_values().tail(10).plot(kind="barh", figsize=(10,5), title="Top 10 Artists by Average Popularity")
    plt.tight_layout(); plt.savefig("top_10_artists_popularity.png", dpi=200); plt.close()
    df.duration_category.value_counts().sort_index().plot(kind="bar", figsize=(8,5), title="Playlist Entries by Duration")
    plt.tight_layout(); plt.savefig("song_duration_distribution.png", dpi=200); plt.close()
    df.is_explicit.map({False:"Non-explicit",True:"Explicit"}).value_counts().plot(kind="bar", figsize=(7,5), title="Explicit vs Non-explicit Entries")
    plt.tight_layout(); plt.savefig("explicit_content_distribution.png", dpi=200); plt.close()
    print("\n".join(summary))
    print("\\nCreated summary, KPI CSVs and four charts.")

if __name__ == "__main__":
    main()
