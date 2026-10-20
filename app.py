import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

st.set_page_config(page_title="US Top 50 Playlist Analysis", layout="wide")
st.title("United States Top 50 Playlist Performance")
st.caption("Historical playlist analytics | 18 May 2024 – 27 November 2025")

@st.cache_data
def load_data():
    d = pd.read_csv("Atlantic_United_States.csv")
    required = ["date","position","song","artist","popularity","duration_ms","album_type","total_tracks","is_explicit"]
    missing = [c for c in required if c not in d.columns]
    if missing: raise ValueError(f"Missing columns: {missing}")
    d["date"] = pd.to_datetime(d["date"], format="%d-%m-%Y", errors="coerce")
    for c in ["position","popularity","duration_ms","total_tracks"]: d[c] = pd.to_numeric(d[c], errors="coerce")
    d["song"] = d.song.astype("string").str.strip()
    d["artist"] = d.artist.astype("string").str.strip().str.replace(r"\s+"," ",regex=True)
    d["album_type"] = d.album_type.astype("string").str.strip().str.lower()
    d["is_explicit"] = d.is_explicit.astype("string").str.lower().str.strip().map({"true":True,"false":False,"1":True,"0":False,"yes":True,"no":False})
    d = d.drop_duplicates().dropna(subset=["date","position","song","artist","popularity"])
    d = d[d.position.between(1,50)].copy()
    d = d.sort_values(["date","song","artist","position"]).drop_duplicates(["date","song","artist"],keep="first")
    d["duration_minutes"] = d.duration_ms / 60000
    d = d.sort_values(["song","artist","date"])
    g = d.groupby(["song","artist"],dropna=False)
    d["days_on_chart"] = g.date.transform("nunique")
    d["average_rank"] = g.position.transform("mean")
    d["best_rank_achieved"] = g.position.transform("min")
    d["rank_volatility_index"] = g.position.transform("std").fillna(0)
    d["popularity_trend_score"] = d.groupby(["song","artist"],dropna=False).popularity.transform(lambda s:s.rolling(7,min_periods=1).mean())
    d["rank_change"] = g.position.diff()
    d["rank_movement"] = d.rank_change.map(lambda x:"First observation" if pd.isna(x) else "Improved" if x<0 else "Declined" if x>0 else "Unchanged")
    d["duration_category"] = pd.cut(d.duration_minutes,[0,3,4,float("inf")],labels=["Short (<3 min)","Medium (3–4 min)","Long (4+ min)"],right=False)
    return d.sort_values(["date","position"]).reset_index(drop=True)

try: df = load_data()
except Exception as e:
    st.error(f"Dataset loading failed: {e}"); st.stop()
if df.empty: st.error("No valid data remains after cleaning."); st.stop()

st.sidebar.header("Filter Playlist Data")
lo,hi = df.date.min().date(),df.date.max().date()
date_range = st.sidebar.date_input("Date range",(lo,hi),min_value=lo,max_value=hi)
artists = sorted(df.artist.dropna().unique().tolist())
chosen_artists = st.sidebar.multiselect("Artist",artists)
pool = df[df.artist.isin(chosen_artists)] if chosen_artists else df
pairs = pool[["song","artist"]].drop_duplicates().sort_values(["artist","song"])
labels = [f"{r.song} — {r.artist}" for r in pairs.itertuples()]
chosen_labels = st.sidebar.multiselect("Song — Artist",labels)
rank = st.sidebar.slider("Rank range",1,50,(1,50))
album_types = sorted(df.album_type.dropna().unique().tolist())
chosen_albums = st.sidebar.multiselect("Album type",album_types,default=album_types)
f = df.copy()
if len(date_range)==2: f=f[f.date.between(pd.Timestamp(date_range[0]),pd.Timestamp(date_range[1]))]
if chosen_artists: f=f[f.artist.isin(chosen_artists)]
if chosen_labels:
    wanted=set(zip(pairs.assign(label=labels).query("label in @chosen_labels").song,pairs.assign(label=labels).query("label in @chosen_labels").artist))
    f=f[f.apply(lambda r:(r.song,r.artist) in wanted,axis=1)]
f=f[f.position.between(*rank) & f.album_type.isin(chosen_albums)]
if f.empty: st.warning("No entries match these filters. Change the selections."); st.stop()

song_kpi=f.groupby(["song","artist"]).agg(days_on_chart=("date","nunique"),average_rank=("position","mean"),best_rank_achieved=("position","min"),rank_volatility_index=("position","std"),popularity_trend_score=("popularity_trend_score","last")).reset_index()
artist_entries=f.groupby("artist").size()
artist_days=f.groupby("artist").date.nunique()
dominance=artist_entries/len(f)*100
cols=st.columns(6)
cols[0].metric("Playlist entries",f"{len(f):,}")
cols[1].metric("Max Days on Chart",f"{song_kpi.days_on_chart.max():,}")
cols[2].metric("Average Rank",f"{f.position.mean():.2f}")
cols[3].metric("Rank Volatility Index",f"{f.position.std():.2f}")
cols[4].metric("Popularity Trend Score",f"{f.popularity_trend_score.mean():.2f}")
cols[5].metric("Explicit Content Share",f"{f.is_explicit.eq(True).mean()*100:.1f}%")
st.caption("Artist Dominance Index = artist share of filtered playlist entries.")

st.subheader("1. Playlist Timeline Explorer")
timeline=f.groupby("date").size().rename("Entries").reset_index()
fig,ax=plt.subplots(figsize=(12,4)); ax.plot(timeline.date,timeline.Entries); ax.set(xlabel="Date",ylabel="Playlist entries",title="Daily Playlist Entries"); ax.grid(alpha=.25); ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2)); ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y")); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)

st.subheader("2. Song Ranking Trends")
opts=sorted(f[["song","artist"]].drop_duplicates().itertuples(index=False,name=None))
opt_labels=[f"{s} — {a}" for s,a in opts]
pick=st.selectbox("Choose song and artist",opt_labels)
s,a=opts[opt_labels.index(pick)]
trend=f[(f.song==s)&(f.artist==a)].sort_values("date")
fig,ax=plt.subplots(figsize=(12,4)); ax.plot(trend.date,trend.position,marker="."); ax.set(xlabel="Date",ylabel="Rank (1 = highest)",title=f"Rank Trend: {s} — {a}"); ax.set_ylim(50,1); ax.grid(alpha=.25); fig.autofmt_xdate(); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
st.dataframe(song_kpi.sort_values("days_on_chart",ascending=False).head(20),use_container_width=True)

st.subheader("3. Artist Dominance Leaderboard")
leader=pd.DataFrame({"Distinct playlist days":artist_days,"Playlist entries":artist_entries,"Artist Dominance Index (%)":dominance}).sort_values("Artist Dominance Index (%)",ascending=False).head(15)
st.dataframe(leader.round(2),use_container_width=True); st.bar_chart(leader["Artist Dominance Index (%)"])

st.subheader("4. Popularity vs Rank")
fig,ax=plt.subplots(figsize=(8,4)); ax.scatter(f.position,f.popularity,alpha=.3,s=18); ax.set(xlabel="Rank (1 = highest)",ylabel="Popularity",title="Popularity vs Playlist Rank"); ax.set_xlim(50,1); ax.set_ylim(0,100); ax.grid(alpha=.2); fig.tight_layout(); st.pyplot(fig); plt.close(fig)
st.write(f"Popularity-position correlation: **{f.popularity.corr(f.position):.3f}**")
bins=pd.cut(f.position,[0,10,20,30,40,50],labels=["1–10","11–20","21–30","31–40","41–50"])
st.write("Popularity by rank range"); st.dataframe(f.assign(rank_range=bins).groupby("rank_range",observed=False).popularity.mean().round(2))

st.subheader("5. Explicit vs Non-explicit Performance")
st.dataframe(f.groupby("is_explicit",dropna=False).agg(entries=("song","size"),average_popularity=("popularity","mean"),average_rank=("position","mean")).round(2),use_container_width=True)

st.subheader("6. Song Entry, Exit and Rank Movement")
sets=f.groupby("date").apply(lambda x:set(zip(x.song,x.artist)),include_groups=False)
rows=[]; prev=set()
for day,cur in sets.items(): rows.append({"date":day,"new_song_entries":len(cur-prev),"song_exits":len(prev-cur)}); prev=cur
st.dataframe(pd.DataFrame(rows),use_container_width=True)
st.write("Rank movement counts"); st.dataframe(f.rank_movement.value_counts().rename_axis("Movement").reset_index(name="Count"))
st.write("Fast risers"); st.dataframe(f.dropna(subset=["rank_change"]).sort_values("rank_change").head(10)[["date","song","artist","rank_change","position"]])
st.write("Slow decliners"); st.dataframe(f.dropna(subset=["rank_change"]).sort_values("rank_change",ascending=False).head(10)[["date","song","artist","rank_change","position"]])

st.subheader("7. Song and Album Attribute Analysis")
st.dataframe(f.groupby("album_type",dropna=False).agg(entries=("song","size"),average_popularity=("popularity","mean"),average_rank=("position","mean"),average_duration_minutes=("duration_minutes","mean"),average_total_tracks=("total_tracks","mean")).round(2),use_container_width=True)
st.dataframe(f.groupby("duration_category",observed=False).agg(entries=("song","size"),average_popularity=("popularity","mean"),average_rank=("position","mean")).round(2),use_container_width=True)
td=f.dropna(subset=["total_tracks"]).copy()
if not td.empty:
    td["track_count_group"]=pd.cut(td.total_tracks,[-1,1,5,10,20,1000],labels=["1 track","2–5 tracks","6–10 tracks","11–20 tracks","21+ tracks"])
    st.write("Total album tracks vs song performance")
    st.dataframe(td.groupby("track_count_group",observed=False).agg(entries=("song","size"),average_popularity=("popularity","mean"),average_rank=("position","mean")).round(2),use_container_width=True)

st.subheader("8. Full Filtered Records")
st.dataframe(f.sort_values(["date","position"]),use_container_width=True)
st.caption("Historical descriptive analysis only; it does not predict future chart positions or establish causation.")
