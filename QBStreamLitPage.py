import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mpld3
import streamlit.components.v1 as components


def check_for_unique_IDs(prices, static):
    for ID in np.unique(np.array(prices['internal_id'])):
        if ID not in np.array(static.index):
            print('Unrecognised ID found: {}'.format(ID))
    else:
        print('All IDS match')
    return None


# we want to deal with any NaNs which fall at the end or start, as these arent interpolated by the interpolate method
def fill_endpoints(dfs):
    for df in dfs:
        df.bfill(inplace=True)
        df.ffill(inplace=True)


def convert_to_USD(dfs):
    for df in dfs:
        df['close USD'] = df['close']*df['fx_to_usd']
        df['mkt_cap_USD'] = df['mkt_cap']*df['fx_to_usd']


def compute_n_day_MA(n, df):
    # for df in dfs:
    df['n-day MA Closing Price'] = df['close USD'].rolling(n).mean()
    # return df


def compute_n_day_STDEV(n, df):
    # for df in dfs:
    df['n-day Close STDEV'] = df['close USD'].rolling(n).std()
    # return df


def add_historical_volatility(df, period):
    """
    This function adds the standard deviation of log returns as a column to the prices dataframe over a period
        Args:
            df (DataFrame) : A Dataframe which contains the time series data
            period (int) : The number of candles which we are calculating the standard deviation of log returns over.
    """
    # for df in dfs:
    close = df.loc[:, "close"]
    log_returns = np.log(close / close.shift(1))
    df.loc[:, 'n-day Vol MA'] = log_returns.rolling(period).std()
    # return df


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
    st.pyplot(fig)

    return fig, axs


def plot_single_bolbands(df):
    compute_n_day_MA(n, df)
    compute_n_day_STDEV(n, df)
    add_historical_volatility(df, n)

    fig = plt.figure()
    fig, axs = plt.subplots(1,1)
    axs.plot(df.index, df['close USD'], label='Close Prices', linewidth=0.7)
    axs.plot(df.index, df['n-day MA Closing Price'], label='n-Day Close MA', linewidth=0.7)
    axs.plot(df.index, df['n-day MA Closing Price'] + df['n-day Close STDEV'] * 1.5,
            label='1.5 STD n-day Upper Band', linewidth=0.5, color='g')
    axs.plot(df.index, df['n-day MA Closing Price'] - df['n-day Close STDEV'] * 1.5,
            label='1.5 STD n-day Lower Band', linewidth=0.5, color='r')
    axs.legend()
    axs.set_title('+/- 1.5STD Bollinger Bands for N Days')
    axs.tick_params(axis='both', which='major', labelsize=8)
    axs.set_xlabel('Date')
    axs.set_ylabel('Close Price USD')

    fig.text(0.5, 0.05, 'Date', ha='center', va='center', fontsize='12')
    fig.text(0.05, 0.5, 'Close Price USD', ha='center', va='center', fontsize='12', rotation='vertical')

    return fig, axs


def currency_volume_MA(n, df, currency_dict):
    vol_curr_dict = {k: {} for k in list(currency_dict.values())}  # Currency symbol is the key. Value is the vol

    for idx, row in df.iterrows():
        date = idx
        currency = row['Raw Currency']

        if date not in list(vol_curr_dict[currency].keys()):
            vol_curr_dict[currency][date] = row['tradable_volume'] * row['close USD']
        else:
            vol_curr_dict[currency][date] += row['tradable_volume'] * row['close USD']

    volume_df = pd.DataFrame(vol_curr_dict)
    vol_non_rolling = volume_df.copy()
    volume_df = volume_df.rolling(n).mean()


    return volume_df, vol_non_rolling


def plot_vol_MA(m, dfs, currency_dict):
    voldf, vol_non_rolling = currency_volume_MA(m, dfs, currency_dict)
    voldf = voldf.dropna(axis=1, how='all')
    vol_non_rolling = vol_non_rolling.dropna(axis=1, how='all')
    fig, axs = plt.subplots(figsize=(36, 36))

    title = list(voldf.columns)[0]
    currency = title
    print(currency)
    date_roll = voldf[currency].dropna().sort_index().index
    date_non_roll = vol_non_rolling[currency].dropna().sort_index().index

    volume_roll = voldf[currency].dropna().sort_index()
    volume_non_roll = vol_non_rolling[currency].dropna().sort_index()

    axs.plot(date_roll, volume_roll, label='n-Day Rolling MA')
    axs.plot(date_non_roll, volume_non_roll, label='Raw Volume')
    axs.set_title(currency, fontsize=50)
    axs.legend(prop={'size': 30})

    axs.set_title(title, fontsize=50)
    axs.tick_params(axis='both', which='major', labelsize=40)
    axs.yaxis.offsetText.set_fontsize(40)

    return fig, axs


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

symbol_idx = {'ADE-NOK':0,'ADE-EUR':1,'BALDB':2,'WDP':3,'DEMANT':4,'OCDO':5,'TKO':6,'TWR':7}

fill_endpoints(df_set)
convert_to_USD(df_set)

header = st.container()

dataCleaning = st.container()

parameter_selector = st.container()

bolband_engine = st.container()

volume_engine = st.container()

st.markdown(
    """
    <style>
    .main{
    background-color: #F5F5F5;
    }
    <style>
    """,
    unsafe_allow_html=True
)

with header:
    st.title('Interactive Time Series Analysis!')

with dataCleaning:
    st.subheader('Data Investigation & Cleaning')
    st.markdown('**First Step Involved Cleaning the Data:**')

    st.markdown(
        """
        - I checked the IDs were consistent between prices and static datasets
        - Found some Zero values; converted these to NaNs, to then:
        - Fill All NaNs via interpolating. There were some NaNs in the first row, and last row, so these were handled by back-filling and forward-filling respectively
        - I noticed we had data for stock ADE in NOK and EUR. I was unsure how to proceed, so created 2 tickers, ADE-NOK and ADE-EUR
        - Also, I believe that the GBp - USA exchange rate was a factor of 100 out, when looking online. I corrected for this by multiplying by 100
        - Replaced internal_id with ticker name for readability, and seperated the data into separate dataframe for each ticker
        """
    )

with parameter_selector:
    st.subheader('Bollinger Band Animation')
    st.markdown('**Choose N, Bollinger Band Period, M, Volume Period, and Ticker Symbol:**')

    N_COL, SYMB_COL = st.columns(2)

    n = N_COL.slider('N Value?', min_value=int(10), max_value=int(120), step=1)
    # m = VOL_PERIOD_COL.slider('M Value?', min_value=int(10), max_value=int(40), step=1)

    symb = SYMB_COL.selectbox('Choose Your Stock:', ['ADE-EUR','ADE-NOK','BALDB','WDP','DEMANT','OCDO','TKO', 'TWR'])
    df = df_set[symbol_idx[symb]]

with bolband_engine:

    fig, ax = plot_single_bolbands(df)
    fig.set_size_inches((8,6))
    fig_html = mpld3.fig_to_html(fig)
    components.html(fig_html, height=650, width=1500)
    st.markdown('**Click Icons Above to Interact with Image**')
    # fig, ax = plot_bol_bands(n, df_set)
    # st.pyplot(fig)

# with volume_engine:
#     fig, ax = plot_vol_MA(m, df, currency_dict)
#     st.pyplot(fig)