from common_imports import *

def download_stock_data(metadata, exp_root_path, start_day, end_day):    
    metadata_df = pd.read_csv(f"{metadata}")
    stock_tickers = list(metadata_df['Ticker'])
    
    # Create the base datasets directory if it doesn't exist
    base_dir = f'./datasets/{exp_root_path}'
    os.makedirs(base_dir, exist_ok=True)
    
    for stock_code in tqdm(stock_tickers, desc="Downloading stock data"):
        try:
            # Get the sector for the current stock
            sector = metadata_df[metadata_df['Ticker'] == stock_code]['Sector'].iloc[0]
            
            # Create sector-specific directory
            sector_dir = os.path.join(base_dir, sector)
            os.makedirs(sector_dir, exist_ok=True)
            
            # Download stock data
            stock_data = pd.DataFrame(fdr.DataReader(stock_code, start_day, end_day))
            stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].astype(float)
            stock_data = stock_data.reset_index()
            
            # Save to CSV in the sector-specific directory
            stock_data.to_csv(os.path.join(sector_dir, f"{stock_code}.csv"), encoding='utf-8', index=False)
            
        except Exception as e:
            print(f"Failed to download data for {stock_code}: {e}")