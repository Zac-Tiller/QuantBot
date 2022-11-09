import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st


'''Functions'''

# Check that the IDs of the prices are matching with IDs of static_data:
# @st.cache
def check_for_unique_IDs(prices, static):
    for ID in np.unique(np.array(prices['internal_id'])):
        if ID not in np.array(static.index):
            print('Unrecognised ID found: {}'.format(ID))
    else:
        print('All IDS match')
    return None


# we want to deal with any NaNs which fall at the end or start, as these arent interpolated by the interpolate method
# @st.cache
def fill_endpoints(dfs):
    for df in dfs:
        df.bfill(inplace=True)
        df.ffill(inplace=True)

# @st.cache
def convert_to_USD(dfs):
    for df in dfs:
        df['close USD'] = df['close']*df['fx_to_usd']
        df['mkt_cap_USD'] = df['mkt_cap']*df['fx_to_usd']

# @st.cache
def compute_n_day_MA(n, dfs):
    for df in dfs:
        df['n-day MA Closing Price'] = df['close USD'].rolling(n).mean()

# @st.cache
def compute_n_day_STDEV(n, dfs):
    for df in dfs:
        df['n-day Close STDEV'] = df['close USD'].rolling(n).std()

# @st.cache
def AddHistoricalVolatility(dfs, period):
    """
    This function adds the standard deviation of log returns as a column to the prices dataframe over a period
        Parameters:
            df (DataFrame) : A Dataframe which contains the time series data
            period (int) : The number of candles which we are calculating the standard deviation of log returns over.
    """
    for df in dfs:
        close = df.loc[:, "close USD"]
        log_returns = np.log(close / close.shift(1))
        df.loc[:, 'n-day Vol MA'] = log_returns.rolling(period).std()

# @st.cache
def plot_bol_bands(n, dfs):
    fig, axs = plt.subplots(4, 2, num=5, figsize=(72, 72))

    titles = [[np.unique(df['internal_id'])[0] for df in dfs[:2]],
              [np.unique(df['internal_id'])[0] for df in dfs[2:4]],
              [np.unique(df['internal_id'])[0] for df in dfs[4:6]],
              [np.unique(df['internal_id'])[0] for df in dfs[6:]]]
    arr = [[0, 1],
           [2, 3],
           [4, 5],
           [6, 7]]

    for j, k in enumerate(axs):
        for column, ax in enumerate(k):
            idx = arr[j][column]
            df = dfs[idx]
            ax.plot(df.index, df['close USD'], label='Close Prices')
            ax.plot(df.index, df['n-day MA Closing Price'], label='n-Day Close MA')
            ax.plot(df.index, df['n-day MA Closing Price'] + df['n-day Close STDEV'] * 1.5,
                    label='1.5 STD n-day Upper Band')
            ax.plot(df.index, df['n-day MA Closing Price'] - df['n-day Close STDEV'] * 1.5,
                    label='1.5 STD n-day Lower Band')
            ax.legend(prop={'size': 30})
            ax.set_title(titles[j][column], fontsize=60)
            ax.tick_params(axis='both', which='major', labelsize=35)

    fig.text(0.5, 0.05, 'Date', ha='center', va='center', fontsize='72')
    fig.text(0.05, 0.5, 'Close Price USD', ha='center', va='center', fontsize='72', rotation='vertical')
    fig.text(0.5, 0.93, 'Bollinger Bands', ha='center', va='center', fontsize='72')

    return fig, axs

# @st.cache
def currency_volume_MA(dfs, currency_dict):
    vol_curr_dict = {k:0 for k in list(currency_dict.values())} # Currency symbol is the key. Value is the vol
    for df in dfs:
        volume_series = df['tradable_volume']*df['close USD']
        volume_USD_MA = volume_series.rolling(21).mean()
        date = df.index
#     for k,v in vol_curr_dict.items():
#         placeholder_df = pd.DataFrame(columns=[k], data=v)
#         placeholder_df['Volume MA'] = placeholder_df[k].rolling(21).mean()
        fig = plt.plot(date, volume_USD_MA, label=df['Raw Currency'][0])
#     fig(figsize=(10,10))
    plt.legend()
    plt.show()


# Import CSV files
prices = pd.read_csv("prices.csv",index_col=0)
static_data = pd.read_csv("static_data.csv",index_col=0)

# create columns for prices data
prices[['close USD', 'mkt_cap_USD']] = 999.99
prices[['n-day Vol MA', 'n-day MA Closing Price', 'n-day Close STDEV']] = 999.99

# convert prices date to a date_time object
prices.index = pd.to_datetime([pd.Timestamp(year=int(str(dt)[:4]), month=int(str(dt)[4:6]), day=int(str(dt)[6:])) for dt in prices.index])
prices.index.name = 'Date'

num_zeros = np.sum(prices.isin([0]).sum(axis=1))
# replace zeros with NaNs so that later we can interpolate
prices.replace({0:float('NaN')}, inplace=True)


static_data.loc[(static_data['symbol']=='ADE') & (static_data['currency']=='EUR'), 'symbol'] = 'ADE_EUR'
static_data.loc[(static_data['symbol']=='ADE') & (static_data['currency']=='NOK'), 'symbol'] = 'ADE_NOK'

symbol_dict = {idx:row['symbol'] for idx, row in static_data.iterrows()}
currency_dict = {row['symbol']:row['currency'] for idx, row in static_data.iterrows()}
prices['internal_id'] = prices['internal_id'].replace(symbol_dict)
prices['Raw Currency'] = prices['internal_id'].replace(currency_dict)

# Define 8 dfs of the prices, and interpolate where we have NaNs
df_BALDB = prices[prices['internal_id']=='BALDB'].interpolate(method='linear')
df_WDP = prices[prices['internal_id']=='WDP'].interpolate(method='linear')
df_DEMANT = prices[prices['internal_id']=='DEMANT'].interpolate(method='linear')
df_OCDO = prices[prices['internal_id']=='OCDO'].interpolate(method='linear')
df_TWR = prices[prices['internal_id']=='TWR'].interpolate(method='linear')
df_TKO = prices[prices['internal_id']=='TKO'].interpolate(method='linear')
df_ADE_EUR = prices[prices['internal_id']=='ADE_EUR'].interpolate(method='linear')
df_ADE_NOK = prices[prices['internal_id']=='ADE_NOK'].interpolate(method='linear')


df_set = [df_ADE_EUR, df_ADE_NOK, df_BALDB, df_WDP, df_DEMANT, df_OCDO, df_TKO, df_TWR]

if __name__ == '__main__':
    fill_endpoints(df_set)
    convert_to_USD(df_set)

    compute_n_day_MA(20, df_set)
    compute_n_day_STDEV(20, df_set)

    AddHistoricalVolatility(df_set, 21)
    plot_bol_bands(20, df_set)

    currency_volume_MA(df_set, currency_dict)




