# streamlit_app.py

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.dates import DateFormatter
import mplcursors
import streamlit as st

# ----- 設定 -----
# スクリプトファイルのディレクトリを基準に data/kanno を参照
BASE_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(BASE_DIR, "gui")

# ディレクトリ存在チェック
if not os.path.isdir(DATA_DIR):
    st.error(f"データディレクトリが見つかりません: {DATA_DIR}")
    st.stop()

# ヘッダー
st.title("FIT Data Interactive Viewer")

# サイドバー：ファイル一覧をラジオボタンで表示
pkls = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".pkl")])
if not pkls:
    st.sidebar.warning("データファイルが見つかりません。ディレクトリを確認してください。")
else:
    selected = st.sidebar.radio("データファイルを選択", pkls, index=0)

    def load_data(filename):
        path = os.path.join(DATA_DIR, filename)
        with open(path, "rb") as f:
            return pickle.load(f)

    def plot_all(data):
        # データ取り出し
        df_segment    = data['df_segment']
        master_df     = data['master_df']
        formatted_dates = data['formatted_dates']
        todays_best   = data['todays_best']
        current_max   = data['current_max']
        mod_remaining = data['mod_remaining']
        name          = data['name']
        formatted_ts0 = data['formatted_ts0']
        durations     = list(range(1, 41))
        dashboard_durations = [1,5,10,20,30,40]

        # Matplotlib Figure
        fig = plt.figure(figsize=(12, 8))
        fig.suptitle(
            f"Data analysis ({name}) {formatted_ts0}", x=0.01, y=0.99,
            ha='left', fontsize=16, fontweight='bold'
        )
        gs = GridSpec(2, 3, figure=fig, width_ratios=[3,1.2,1.2],
                      hspace=0.15, wspace=0.8)

        # (1) Power & MOD
        ax0 = fig.add_subplot(gs[0,0])
        ax0.plot(df_segment.index, df_segment['power'], lw=2, label='Power [W]')
        ax0.set_ylabel("Power [W]")
        ax0.grid(ls='--', alpha=0.5)
        ax0.set_xticklabels([])
        ax0b = ax0.twinx()
        ax0b.plot(df_segment.index, mod_remaining, color='purple', lw=2, label='MOD Remaining [J]')
        ax0b.set_ylabel("MOD [J]")
        ax0b.set_ylim(bottom=0)
        h1,l1 = ax0.get_legend_handles_labels()
        h2,l2 = ax0b.get_legend_handles_labels()
        ax0.legend(h1+h2, l1+l2, loc='upper left')

        # (2) Speed & Predicted + 状態背景色 + cadence
        ax2 = fig.add_subplot(gs[1,0], sharex=ax0)
        ax2.plot(df_segment.index, df_segment['speed'], color='tab:blue', lw=2)
        ax2.plot(df_segment.index, df_segment['pred_speed'], color='tab:orange', ls='--', lw=2)
        for status, color in [('deceleration','blue'),('acceleration','red')]:
            cond = df_segment['status']==status
            groups = (cond != cond.shift()).cumsum()
            for _, grp in df_segment[cond].groupby(groups):
                ax2.axvspan(grp.index[0], grp.index[-1], color=color, alpha=0.15)
        ax2.set_ylabel("Speed [km/h]")
        ax2.grid(ls='--', alpha=0.5)
        ax2.set_xlabel("Time")
        ax2.xaxis.set_major_formatter(DateFormatter("%H:%M:%S"))
        ax2b = ax2.twinx()
        ax2b.plot(df_segment.index, df_segment['cadence'], color='green', lw=2)
        ax2b.set_ylabel("Cadence [rpm]")
        from matplotlib.lines import Line2D
        legend_elems = [
            Line2D([0],[0],color='tab:blue', lw=2, label='Actual Speed'),
            Line2D([0],[0],color='tab:orange', ls='--', lw=2, label='Predicted Speed'),
            Line2D([0],[0],color='green', lw=2, label='Cadence [rpm]'),
        ]
        ax2.legend(handles=legend_elems, loc='upper left')

        # (3) Best Dashboard
        ax_dash = fig.add_subplot(gs[0,1])
        ax_dash.axis('off')
        lines = [f"{d:>2}s: {master_df.at[d,'max_power']:>6.1f} W ({formatted_dates[d]})"
                 for d in dashboard_durations]
        ax_dash.text(0,1, "Best Dashboard\n\n"+"\n".join(lines),
                     va='top', ha='left', fontsize=12, family='monospace',
                     bbox=dict(boxstyle='round,pad=1', fc='#f9f9f9', ec='gray', alpha=0.7))

        # (4) Today's Dashboard
        ax_today = fig.add_subplot(gs[0,2])
        ax_today.axis('off')
        today_lines = [f"{d:>2}s: {todays_best[d]:>6.1f} W"
                       for d in dashboard_durations]
        ax_today.text(0,1, "Today's Dashboard\n\n"+"\n".join(today_lines),
                      va='top', ha='left', fontsize=12, family='monospace',
                      bbox=dict(boxstyle='round,pad=1', fc='#fffbe6', ec='gray', alpha=0.8))

        # (5) Power–Duration Curve
        ax_curve = fig.add_subplot(gs[1,1:3])
        baseline = [master_df.at[d,'max_power'] for d in durations]
        ax_curve.plot(durations, [current_max[d] for d in durations], '-o', label='Analysis PD')
        ax_curve.plot(durations, baseline, '-s', label='Best PD')
        for i,d in enumerate(durations):
            if baseline[i] < current_max[d]:
                ax_curve.plot(d, baseline[i], 's', ms=8, color='red')
        ax_curve.set_xticks(durations)
        ax_curve.set_xlabel("Duration [s]")
        ax_curve.set_ylabel("Max Avg Power [W]")
        ax_curve.set_title("Power Duration Curve")
        ax_curve.grid(ls='--', alpha=0.5)
        ax_curve.legend(loc='upper right')

        fig.subplots_adjust(top=0.92, left=0.05, right=0.98, bottom=0.06)

        # ホバーで値を表示
        for ax in fig.axes:
            mplcursors.cursor(ax, hover=True).connect(
                'add', lambda sel: sel.annotation.set_text(
                    f"{sel.target[0]:%H:%M:%S}\n{sel.target[1]:.2f}"))

        st.pyplot(fig)

    # 選択されたファイルを描画
    try:
        data = load_data(selected)
        if isinstance(data, pd.DataFrame):
            st.error("Data format error: please run prepare_gui_data.py before gui_app.py")
        else:
            plot_all(data)
    except Exception as e:
        st.error(f"Failed to load GUI data: {e}")
