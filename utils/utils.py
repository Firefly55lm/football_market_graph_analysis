import pandas as pd
import numpy as np
import community
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx


def get_capital_gain_df(df: pd.DataFrame, filters: list = ["is_free", "is_loan", "is_loan_end", "is_retired"],
                        columns_to_keep:list = ['team_name','counter_team_name', 'counter_team_country','player_id', 'transfer_fee_amnt', 'season', 'season_ord', 'window'],
                        columns_to_fill:list = {"transfer_fee_amnt": 0}, season_column:str = "season", window_column:str = "window",
                        transfer_fee_column:str = "transfer_fee_amnt"):
    
    for filter in filters:
        df = df[df[filter] == False]

    for column in columns_to_fill.keys():
        df[column] = df[column].fillna(columns_to_fill[column])

    df["season_ord"] = df.apply(lambda row: row[season_column] + 0.1 if row[window_column]=="w" else row[season_column] + 0.2, axis=1)

    in_tr = df[df["dir"]=="in"][columns_to_keep]
    left_tr = df[df["dir"]=="left"][columns_to_keep]

    in_tr = in_tr.rename(columns={transfer_fee_column: "arrival_fee_amnt"})

    merged_df = pd.merge(left_tr, in_tr, on=['team_name', 'player_id'], how='left', suffixes=["_left", "_in"])
    merged_df = merged_df.dropna()
    merged_df = merged_df[merged_df["season_ord_left"]>merged_df["season_ord_in"]]
    merged_df["gain"] = merged_df[transfer_fee_column] - merged_df["arrival_fee_amnt"]

    return merged_df



def get_capital_gain_df_wagents(df: pd.DataFrame, filters: list = ["is_free", "is_loan", "is_loan_end", "is_retired"],
                        columns_to_keep:list = ['team_name','counter_team_name', 'counter_team_country', 'agent', 'player_id', 'transfer_fee_amnt', 'season', 'season_ord', 'window'],
                        columns_to_fill:list = {"transfer_fee_amnt": 0}, season_column:str = "season", window_column:str = "window",
                        transfer_fee_column:str = "transfer_fee_amnt"):
    
    for filter in filters:
        df = df[df[filter] == False]

    for column in columns_to_fill.keys():
        df[column] = df[column].fillna(columns_to_fill[column])

    df["season_ord"] = df.apply(lambda row: row[season_column] + 0.1 if row[window_column]=="w" else row[season_column] + 0.2, axis=1)

    in_tr = df[df["dir"]=="in"][columns_to_keep]
    left_tr = df[df["dir"]=="left"][columns_to_keep]

    in_tr = in_tr.rename(columns={transfer_fee_column: "arrival_fee_amnt"})

    merged_df = pd.merge(left_tr, in_tr, on=['team_name', 'player_id'], how='left', suffixes=["_left", "_in"])
    merged_df = merged_df.dropna()
    merged_df = merged_df[merged_df["season_ord_left"]>merged_df["season_ord_in"]]
    merged_df["gain"] = merged_df[transfer_fee_column] - merged_df["arrival_fee_amnt"]

    return merged_df



def load_theme(theme_name:str) -> dict:
        """
        Loads the Seaborn/Matplolib theme.
        -----
        Args:
            * theme_name: the name of the theme ("dark" or "light")
        --------
        Returns:
            * A dictionary with the theme's information.
        """

        dark = {
            "figure.facecolor": "#202021",
            "axes.facecolor": "#262626",
            "axes.edgecolor": "#cfcfd1",
            "axes.grid": True,
            "grid.color": "#555555",
            "grid.linewidth": 0.5,
            "xtick.color": "#ffffff",
            "ytick.color": "#ffffff",
            "axes.labelcolor": "#ffffff"
            }
        
        light = {
            "figure.facecolor": "#ffffff",
            "axes.facecolor": "#303030",
            "axes.edgecolor": "#171717",
            "axes.grid": True,
            "grid.color": "#555555",
            "grid.linewidth": 0.5,
            "xtick.color": "#000000",
            "ytick.color": "#000000",
            "axes.labelcolor": "#000000"
            }
        
        if theme_name == "dark":
            sns.set_style("dark", rc=dark)
            return dark
        elif theme_name == "light":
            sns.set_style("dark", rc=light)
            return light
        else:
            raise ValueError("\033[31mPlease provide a valid theme name.\033[0m")


def cohesion_metrics(graph, ci_as_mean=False):
    clustering_individual = nx.clustering(graph)
    sorted_ci = sorted(clustering_individual.items(), key=lambda x:x[1], reverse=True)[:10]
    if ci_as_mean is True:
        sorted_ci = np.mean([value for (_, value) in sorted_ci])
    transitivity = nx.transitivity(graph)
    density = nx.density(graph)
    assortativity = nx.degree_assortativity_coefficient(graph)
    g_undirected = graph.to_undirected()
    partition_g = community.best_partition(g_undirected)
    modularity = community.modularity(partition_g, g_undirected)
    return {"transitivity": transitivity, "density": density, "assortativity": assortativity, "modularity": modularity, "clustering": sorted_ci}


if __name__ == "__main__":
    df = pd.read_csv('dataset/transfers_complete.csv')
    
    cg_df = get_capital_gain_df(df)
    print(cg_df.head(20))
    print(cg_df.keys())