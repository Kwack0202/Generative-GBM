from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.visualization import *
from utils.metric import *

class Backtesting:
    def __init__(self, args):
        self.args = args
        self.years = [2022, 2023, 2024]
   
    def UpDown_Signal(self):
        # 기본 경로 설정
        base_input_path = f"./stock_prediction/pred_results/{self.args.model_name}/"
        base_output_path = "./stock_prediction/up_down_signal/"
                
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 메소드 방식 목록
        methods = ["origin_data", "Diffusion", "GAN", "Mathematical"]
        
        for method in methods:
            print(f"[INFO] Processing method: {method}")
            
            # 입력 경로 설정
            method_input_path = os.path.join(base_input_path, method)
            
            # origin_data의 경우 세부 모델명 없이 바로 time_series_length 후 exp_root_path로 이동
            if method == "origin_data":
                input_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path, self.args.sector)
                # 출력 경로: origin_data은 method 폴더 아래 저장, Scenarios 폴더 제외
                output_path = os.path.join(base_output_path, self.args.model_name, method, time_series_length, self.args.exp_root_path, self.args.sector)
                subdirs = [""]
            else:
                # Diffusion, GAN, Mathematical은 세부 모델명 폴더 탐색
                subdirs = [d for d in os.listdir(method_input_path) if os.path.isdir(os.path.join(method_input_path, d))]
                if not subdirs:
                    print(f"[WARNING] No submodel directories found for {method}. Skipping.")
                    continue
            
            for subdir in subdirs:
                if method != "origin_data":
                    # 입력 경로: 세부 모델명 포함
                    input_path = os.path.join(method_input_path, subdir, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                    # 출력 경로: 알고리즘 방식(method)을 건너뛰고 세부 모델명(subdir)으로 저장
                    output_path = os.path.join(base_output_path, self.args.model_name, subdir, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                
                if not os.path.exists(input_path):
                    print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                    continue
                
                # 섹터 내 모든 종목(폴더) 탐색
                tickers = [d for d in os.listdir(input_path) if os.path.isdir(os.path.join(input_path, d))]
                if not tickers:
                    print(f"[WARNING] No tickers found in {input_path}. Skipping.")
                    continue
                
                for ticker in tickers:
                    ticker_path = os.path.join(input_path, ticker)
                    # window_1.csv ~ window_3.csv 병합
                    merged_data = []
                    for window in range(1, 4):
                        window_file = os.path.join(ticker_path, f"window_{window}.csv")
                        if os.path.exists(window_file):
                            df = pd.read_csv(window_file)
                            mean_predicted = df['Predicted'].median()
                            df['Predicted'] = df['Predicted'].apply(lambda x: 1 if x > mean_predicted else 0)
                            merged_data.append(df)
                        else:
                            print(f"[WARNING] {window_file} does not exist.")
                    
                    if not merged_data:
                        print(f"[WARNING] No data found for {ticker} in {ticker_path}. Skipping.")
                        continue
                    
                    # 데이터 병합
                    merged_df = pd.concat(merged_data, ignore_index=True)
                    # 열 순서 정리: Actual, Predicted
                    merged_df = merged_df[['Actual', 'Predicted']]
                    
                    # 출력 경로 생성 및 저장
                    output_file = os.path.join(output_path, f"{ticker}.csv")
                    os.makedirs(output_path, exist_ok=True)
                    merged_df.to_csv(output_file, index=False)
                    print(f"[INFO] Saved merged results for {ticker} to {output_file}")
                                   
    def BuySell_Signal(self):
        # 기본 경로 설정
        base_input_path = f"./stock_prediction/up_down_signal/{self.args.model_name}/"
        base_output_path = "./stock_prediction/individual_trading/buy_sell_signal/"
        data_path = f"./datasets/{self.args.exp_root_path}/{self.args.sector}/"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
        
        opposite_count = 1
        split_num = 753
        
        for model in tqdm(model_types, desc="Processing model types"):           
            # origin_data의 경우 Scenarios 폴더 없이 바로 exp_root_path
            if model == "origin_data":
                input_path = os.path.join(base_input_path, model, time_series_length, self.args.exp_root_path, self.args.sector)
                output_path = os.path.join(base_output_path, 'Full_period', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
            else:
                input_path = os.path.join(base_input_path, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                output_path = os.path.join(base_output_path, 'Full_period', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
            
            if not os.path.exists(input_path):
                print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                continue
            
            # 섹터 내 모든 종목(파일) 탐색
            tickers = [f[:-4] for f in os.listdir(data_path) if f.endswith('.csv')]
            if not tickers:
                print(f"[WARNING] No tickers found in {data_path}. Skipping.")
                continue
            
            for ticker in tickers:
                # 원본 주식 데이터 로드
                stock_file = os.path.join(data_path, f"{ticker}.csv")
                if not os.path.exists(stock_file):
                    print(f"[WARNING] Stock data {stock_file} does not exist. Skipping.")
                    continue
                
                stock_data = pd.read_csv(stock_file).iloc[-split_num:, :6].reset_index(drop=True)
                
                # 예측 결과 로드
                result_file = os.path.join(input_path, f"{ticker}.csv")
                if not os.path.exists(result_file):
                    print(f"[WARNING] Result file {result_file} does not exist. Skipping.")
                    continue
                
                model_results_data = pd.read_csv(result_file).reset_index(drop=True)
                
                # 데이터 결합
                trading_data = pd.concat([stock_data, model_results_data], axis=1)
                
                # 거래 신호 생성
                action = "No action"
                counter = 0
                initial_position_set = False
                
                for i in range(len(trading_data)):
                    curr_pos = trading_data.loc[i, 'Predicted']
                    prev_pos = trading_data.loc[i-1, 'Predicted'] if i > 0 else 0
                    
                    if not initial_position_set:
                        if curr_pos == 0:
                            action = "No action"
                        else:
                            action = "Buy"
                            initial_position_set = True
                    else:
                        last_action = trading_data.loc[i-1, 'action'] if i > 0 else "No action"
                        
                        if last_action == "Sell":
                            if curr_pos == 0:
                                action = "No action"
                                initial_position_set = False
                            else:
                                action = "Buy"
                                counter = 0
                        else:
                            if curr_pos == 1:
                                action = "Holding"
                                counter = 0
                            else:
                                counter += 1
                                if counter == opposite_count:
                                    action = "Sell"
                                    counter = 0
                                else:
                                    action = "Holding"
                    
                    # 마지막 행 처리: 이전이 Holding 또는 Buy인 경우에만 sell
                    if i == len(trading_data) - 1:
                        last_action = trading_data.loc[i-1, 'action'] if i > 0 else "No action"
                        if last_action in ["Holding", "Buy"]:
                            action = "Sell"
                        else:
                            action = "No action"
                    
                    trading_data.loc[i, 'action'] = action
                
                # 출력 저장
                output_file = os.path.join(output_path, f"{ticker}.csv")
                os.makedirs(output_path, exist_ok=True)
                trading_data.to_csv(output_file, index=True)
                print(f"[INFO] Saved buy/sell signals for {ticker} to {output_file}")
    
    def BuySell_Signal_YOY(self):
        # 기본 경로 설정
        base_input_path = f"./stock_prediction/up_down_signal/{self.args.model_name}/"
        base_output_path = "./stock_prediction/individual_trading/buy_sell_signal/"
        data_path = f"./datasets/{self.args.exp_root_path}/{self.args.sector}/"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
        
        opposite_count = 1
        split_num = 753
        
        for year in self.years:
            for model in tqdm(model_types, desc="Processing model types"):                
                # origin_data의 경우 Scenarios 폴더 없이 바로 exp_root_path
                if model == "origin_data":
                    input_path = os.path.join(base_input_path, model, time_series_length, self.args.exp_root_path, self.args.sector)
                    output_path = os.path.join(base_output_path, f'Year_{year}', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
                else:
                    input_path = os.path.join(base_input_path, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                    output_path = os.path.join(base_output_path, f'Year_{year}', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                
                if not os.path.exists(input_path):
                    print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                    continue
                
                # 섹터 내 모든 종목(파일) 탐색
                tickers = [f[:-4] for f in os.listdir(data_path) if f.endswith('.csv')]
                if not tickers:
                    print(f"[WARNING] No tickers found in {data_path}. Skipping.")
                    continue
                
                for ticker in tickers:
                    # 원본 주식 데이터 로드
                    stock_file = os.path.join(data_path, f"{ticker}.csv")
                    if not os.path.exists(stock_file):
                        print(f"[WARNING] Stock data {stock_file} does not exist. Skipping.")
                        continue
                    
                    stock_data = pd.read_csv(stock_file).iloc[-split_num:, :6].reset_index(drop=True)
                    
                    # 예측 결과 로드
                    result_file = os.path.join(input_path, f"{ticker}.csv")
                    if not os.path.exists(result_file):
                        print(f"[WARNING] Result file {result_file} does not exist. Skipping.")
                        continue
                    
                    model_results_data = pd.read_csv(result_file).reset_index(drop=True)
                    
                    # 데이터 결합
                    trading_data = pd.concat([stock_data, model_results_data], axis=1)
                    trading_data['index'] = pd.to_datetime(trading_data['index'])
                    trading_data = trading_data[trading_data['index'].dt.year == year].reset_index(drop=True)
                    
                    # 거래 신호 생성
                    action = "No action"
                    counter = 0
                    initial_position_set = False
                    
                    for i in range(len(trading_data)):
                        curr_pos = trading_data.loc[i, 'Predicted']
                        prev_pos = trading_data.loc[i-1, 'Predicted'] if i > 0 else 0
                        
                        if not initial_position_set:
                            if curr_pos == 0:
                                action = "No action"
                            else:
                                action = "Buy"
                                initial_position_set = True
                        else:
                            last_action = trading_data.loc[i-1, 'action'] if i > 0 else "No action"
                            
                            if last_action == "Sell":
                                if curr_pos == 0:
                                    action = "No action"
                                    initial_position_set = False
                                else:
                                    action = "Buy"
                                    counter = 0
                            else:
                                if curr_pos == 1:
                                    action = "Holding"
                                    counter = 0
                                else:
                                    counter += 1
                                    if counter == opposite_count:
                                        action = "Sell"
                                        counter = 0
                                    else:
                                        action = "Holding"
                        
                        # 마지막 행 처리: 이전이 Holding 또는 Buy인 경우에만 sell
                        if i == len(trading_data) - 1:
                            last_action = trading_data.loc[i-1, 'action'] if i > 0 else "No action"
                            if last_action in ["Holding", "Buy"]:
                                action = "Sell"
                            else:
                                action = "No action"
                        
                        trading_data.loc[i, 'action'] = action
                    
                    # 출력 저장
                    output_file = os.path.join(output_path, f"{ticker}.csv")
                    os.makedirs(output_path, exist_ok=True)
                    trading_data.to_csv(output_file, index=True)
                    print(f"[INFO] Saved buy/sell signals for {ticker} to {output_file}")            
    
    def Simulation(self):
        # 기본 경로 설정
        up_down_path = f"./stock_prediction/up_down_signal/{self.args.model_name}/"  

        base_input_path = "./stock_prediction/individual_trading/buy_sell_signal/"
        base_output_path = "./stock_prediction/individual_trading/simulation/"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(up_down_path) if os.path.isdir(os.path.join(up_down_path, d))]
        
        commission_rate = 0.0005
        
        for model in tqdm(model_types, desc="Processing model types"):
            # 입력/출력 경로 설정
            if model == "origin_data":
                input_path = os.path.join(base_input_path, 'Full_period', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
                output_path = os.path.join(base_output_path, 'Full_period', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
            else:
                input_path = os.path.join(base_input_path, 'Full_period', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                output_path = os.path.join(base_output_path, 'Full_period', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
            
            if not os.path.exists(input_path):
                print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                continue
            
            # 섹터 내 모든 종목 탐색
            tickers = [f[:-4] for f in os.listdir(input_path) if f.endswith('.csv')]
            if not tickers:
                print(f"[WARNING] No tickers found in {input_path}. Skipping.")
                continue
            
            for ticker in tickers:
                # 거래 신호 데이터 로드
                signal_file = os.path.join(input_path, f"{ticker}.csv")
                if not os.path.exists(signal_file):
                    print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                    continue
                
                df = pd.read_csv(signal_file, index_col=0)
                
                # 새로운 데이터프레임 생성
                new_data = {
                    'index': [],  # Date 대신 index 사용
                    'Margin_Profit': [],
                    'Margin_Return': [],
                    'Cumulative_Profit': [],
                }
                
                buy_price = None
                cumulative_profit = 0
                
                for index, row in df.iterrows():
                    if row['action'] == 'Buy':
                        buy_price = row['Close']
                        new_data['index'].append(row['index'])
                        new_data['Margin_Profit'].append(0)
                        new_data['Cumulative_Profit'].append(cumulative_profit)
                        new_data['Margin_Return'].append(0)
                    
                    elif row['action'] == 'Sell' and buy_price is not None:
                        sell_price = row['Close']
                        profit = sell_price - buy_price - (sell_price * commission_rate)
                        return_ = (profit / buy_price) * 100 if buy_price != 0 else 0
                        cumulative_profit += profit
                        
                        new_data['index'].append(row['index'])
                        new_data['Margin_Profit'].append(profit)
                        new_data['Cumulative_Profit'].append(cumulative_profit)
                        new_data['Margin_Return'].append(return_)
                    
                    else:
                        new_data['index'].append(row['index'])
                        new_data['Margin_Profit'].append(0)
                        new_data['Cumulative_Profit'].append(cumulative_profit)
                        new_data['Margin_Return'].append(0)
                
                # 새로운 데이터프레임 생성
                new_df = pd.DataFrame(new_data)
                
                # 원본 데이터와 병합
                merged_df = pd.merge(df, new_df, on='index', how='outer')
                
                # Holding_Period 계산
                merged_df['Holding_Period'] = merged_df.groupby((merged_df['action'] != 'Holding').cumsum()).cumcount()
                
                # 초기 투자 계산
                position_mask = (merged_df['action'] == 'Buy')
                if position_mask.sum() == 0:
                    initial_investment = merged_df.iloc[0]['Close']
                else:
                    buy_index = merged_df[position_mask].index[0]
                    initial_investment = merged_df.iloc[buy_index + 1]['Open'] if buy_index + 1 < len(merged_df) else merged_df.iloc[buy_index]['Close']
                
                # Cumulative_Return 계산
                merged_df['Cumulative_Return'] = (merged_df['Cumulative_Profit'] / initial_investment) * 100
                
                # Drawdown 계산
                merged_df['Drawdown'] = 0.0
                merged_df['Drawdown_rate'] = 0.0
                peak_profit = merged_df['Cumulative_Profit'].iloc[0]
                peak_profit_rate = merged_df['Cumulative_Return'].iloc[0]
                
                for index, row in merged_df.iterrows():
                    current_profit = row['Cumulative_Profit']
                    if current_profit > peak_profit:
                        peak_profit = current_profit
                    drawdown = -(peak_profit - current_profit)
                    merged_df.at[index, 'Drawdown'] = drawdown
                    
                    current_profit_rate = row['Cumulative_Return']
                    if current_profit_rate > peak_profit_rate:
                        peak_profit_rate = current_profit_rate
                    drawdown_rate = -(peak_profit_rate - current_profit_rate)
                    merged_df.at[index, 'Drawdown_rate'] = drawdown_rate
                
                # 열 순서 지정
                column_names = [
                    "index", "Open", "High", "Low", "Close", "Predicted", "action",
                    "Margin_Profit", "Cumulative_Profit", "Margin_Return", "Cumulative_Return",
                    "Drawdown", "Drawdown_rate", "Holding_Period"
                ]
                merged_df = merged_df[column_names]
                
                # 숫자형 변수 반올림
                merged_df = merged_df.round(3)
                
                # 출력 저장
                output_file = os.path.join(output_path, f"{ticker}.csv")
                os.makedirs(output_path, exist_ok=True)
                merged_df.to_csv(output_file, index=True)
                print(f"[INFO] Saved simulation results for {ticker} to {output_file}")
    
    def Simulation_YOY(self):
        # 기본 경로 설정
        up_down_path = f"./stock_prediction/up_down_signal/{self.args.model_name}/"  

        base_input_path = "./stock_prediction/individual_trading/buy_sell_signal/"
        base_output_path = "./stock_prediction/individual_trading/simulation/"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(up_down_path) if os.path.isdir(os.path.join(up_down_path, d))]
        
        commission_rate = 0.0005
        
        for year in self.years:
            for model in tqdm(model_types, desc="Processing model types"):
                # 입력/출력 경로 설정
                if model == "origin_data":
                    input_path = os.path.join(base_input_path, f'Year_{year}', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
                    output_path = os.path.join(base_output_path, f'Year_{year}', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
                else:
                    input_path = os.path.join(base_input_path, f'Year_{year}', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                    output_path = os.path.join(base_output_path, f'Year_{year}', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                
                if not os.path.exists(input_path):
                    print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                    continue
                
                # 섹터 내 모든 종목 탐색
                tickers = [f[:-4] for f in os.listdir(input_path) if f.endswith('.csv')]
                if not tickers:
                    print(f"[WARNING] No tickers found in {input_path}. Skipping.")
                    continue
                
                for ticker in tickers:
                    # 거래 신호 데이터 로드
                    signal_file = os.path.join(input_path, f"{ticker}.csv")
                    if not os.path.exists(signal_file):
                        print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                        continue
                    
                    df = pd.read_csv(signal_file, index_col=0)
                    
                    # 새로운 데이터프레임 생성
                    new_data = {
                        'index': [],  # Date 대신 index 사용
                        'Margin_Profit': [],
                        'Margin_Return': [],
                        'Cumulative_Profit': [],
                    }
                    
                    buy_price = None
                    cumulative_profit = 0
                    
                    for index, row in df.iterrows():
                        if row['action'] == 'Buy':
                            buy_price = row['Close']
                            new_data['index'].append(row['index'])
                            new_data['Margin_Profit'].append(0)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(0)
                        
                        elif row['action'] == 'Sell' and buy_price is not None:
                            sell_price = row['Close']
                            profit = sell_price - buy_price - (sell_price * commission_rate)
                            return_ = (profit / buy_price) * 100 if buy_price != 0 else 0
                            cumulative_profit += profit
                            
                            new_data['index'].append(row['index'])
                            new_data['Margin_Profit'].append(profit)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(return_)
                        
                        else:
                            new_data['index'].append(row['index'])
                            new_data['Margin_Profit'].append(0)
                            new_data['Cumulative_Profit'].append(cumulative_profit)
                            new_data['Margin_Return'].append(0)
                    
                    # 새로운 데이터프레임 생성
                    new_df = pd.DataFrame(new_data)
                    
                    # 원본 데이터와 병합
                    merged_df = pd.merge(df, new_df, on='index', how='outer')
                    
                    # Holding_Period 계산
                    merged_df['Holding_Period'] = merged_df.groupby((merged_df['action'] != 'Holding').cumsum()).cumcount()
                    
                    # 초기 투자 계산
                    position_mask = (merged_df['action'] == 'Buy')
                    if position_mask.sum() == 0:
                        initial_investment = merged_df.iloc[0]['Close']
                    else:
                        buy_index = merged_df[position_mask].index[0]
                        initial_investment = merged_df.iloc[buy_index + 1]['Open'] if buy_index + 1 < len(merged_df) else merged_df.iloc[buy_index]['Close']
                    
                    # Cumulative_Return 계산
                    merged_df['Cumulative_Return'] = (merged_df['Cumulative_Profit'] / initial_investment) * 100
                    
                    # Drawdown 계산
                    merged_df['Drawdown'] = 0.0
                    merged_df['Drawdown_rate'] = 0.0
                    peak_profit = merged_df['Cumulative_Profit'].iloc[0]
                    peak_profit_rate = merged_df['Cumulative_Return'].iloc[0]
                    
                    for index, row in merged_df.iterrows():
                        current_profit = row['Cumulative_Profit']
                        if current_profit > peak_profit:
                            peak_profit = current_profit
                        drawdown = -(peak_profit - current_profit)
                        merged_df.at[index, 'Drawdown'] = drawdown
                        
                        current_profit_rate = row['Cumulative_Return']
                        if current_profit_rate > peak_profit_rate:
                            peak_profit_rate = current_profit_rate
                        drawdown_rate = -(peak_profit_rate - current_profit_rate)
                        merged_df.at[index, 'Drawdown_rate'] = drawdown_rate
                    
                    # 열 순서 지정
                    column_names = [
                        "index", "Open", "High", "Low", "Close", "Predicted", "action",
                        "Margin_Profit", "Cumulative_Profit", "Margin_Return", "Cumulative_Return",
                        "Drawdown", "Drawdown_rate", "Holding_Period"
                    ]
                    merged_df = merged_df[column_names]
                    
                    # 숫자형 변수 반올림
                    merged_df = merged_df.round(3)
                    
                    # 출력 저장
                    output_file = os.path.join(output_path, f"{ticker}.csv")
                    os.makedirs(output_path, exist_ok=True)
                    merged_df.to_csv(output_file, index=True)
                    print(f"[INFO] Saved simulation results for {ticker} to {output_file}")
    
    def BacktestSummary(self):
        # 기본 경로 설정
        up_down_path = f"./stock_prediction/up_down_signal/{self.args.model_name}/"          
        
        base_input_path = "./stock_prediction/individual_trading/simulation/"
        base_output_path = "./stock_prediction/individual_trading/backtesting/"
                
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(up_down_path) if os.path.isdir(os.path.join(up_down_path, d))]
        
        summary_data = []
        
        for model in tqdm(model_types, desc="Processing model types"):
            if model == "origin_data":
                input_path = os.path.join(base_input_path, 'Full_period', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
                scenarios = 0
            else:
                input_path = os.path.join(base_input_path, 'Full_period', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                scenarios = self.args.num_scenarios
            
            if not os.path.exists(input_path):
                print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                continue
            
            tickers = [f[:-4] for f in os.listdir(input_path) if f.endswith('.csv')]
            if not tickers:
                print(f"[WARNING] No tickers found in {input_path}. Skipping.")
                continue
            
            for ticker in tickers:
                signal_file = os.path.join(input_path, f"{ticker}.csv")
                if not os.path.exists(signal_file):
                    print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                    continue
                
                df = pd.read_csv(signal_file, index_col=0)
                
                df['action'] = df['action'].replace({'No action': 0, 'Buy': 1, 'Sell': -1})
                
                no_trade = len(df[df['Margin_Profit'] > 0]) + len(df[df['Margin_Profit'] < 0])
                
                max_holding_period = df[df['action'] == 'Holding']['Holding_Period'].max() if no_trade > 0 else 0
                max_holding_period = 0 if pd.isna(max_holding_period) else max_holding_period
                
                mean_holding_period = df[df['action'] == 'Holding']['Holding_Period'].mean() if no_trade > 0 else 0
                mean_holding_period = 0 if pd.isna(mean_holding_period) else mean_holding_period
                
                winning_ratio = len(df[(df['action'] == -1) & (df['Margin_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                
                profit_average = df[df['Margin_Profit'] > 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] > 0]) > 0 else 0
                profit_average = 0 if pd.isna(profit_average) else profit_average
                loss_average = df[df['Margin_Profit'] < 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] < 0]) > 0 else 0
                loss_average = 0 if pd.isna(loss_average) else loss_average
                
                payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                loss_sum = df[df['Margin_Profit'] < 0]['Margin_Profit'].sum()
                profit_sum = df[df['Margin_Profit'] > 0]['Margin_Profit'].sum()
                profit_factor = -profit_sum / loss_sum if loss_sum < 0 else 0
                
                final_cumulative_profit = df['Cumulative_Profit'].iloc[-1]
                final_cumulative_return = df['Cumulative_Return'].iloc[-1]
                
                max_realized_profit = df['Margin_Profit'].max() if no_trade > 0 else 0
                max_realized_profit = 0 if pd.isna(max_realized_profit) else max_realized_profit
                max_realized_return = df['Margin_Return'].max() if no_trade > 0 else 0
                max_realized_return = 0 if pd.isna(max_realized_return) else max_realized_return
                
                MDD = df['Drawdown'].min() if no_trade > 0 else 0
                MDD = 0 if pd.isna(MDD) else MDD
                MDD_rate = df['Drawdown_rate'].min() if no_trade > 0 else 0
                MDD_rate = 0 if pd.isna(MDD_rate) else MDD_rate
                
                summary_data.append([
                    "Full_period", model, self.args.sector, ticker, time_series_length, scenarios,
                    no_trade, max_holding_period, mean_holding_period, winning_ratio,
                    profit_average, loss_average, payoff_ratio, profit_factor,
                    final_cumulative_profit, final_cumulative_return,
                    max_realized_profit, max_realized_return,
                    MDD, MDD_rate
                ])
        
        summary_df = pd.DataFrame(summary_data, columns=[
            "year", "model", "sector", "ticker", "train_test_length", "scenarios",
            "no_trade", "max_holding_period", "mean_holding_period", "winning_ratio",
            "profit_average", "loss_average", "payoff_ratio", "profit_factor",
            "final_cumulative_profit", "final_cumulative_return",
            "max_realized_profit", "max_realized_return",
            "MaxDrawdown", "MaxDrawdown_rate"
        ])
        
        summary_df = summary_df.round(3)
        
        output_dir = os.path.join(base_output_path, "Full_period")
        os.makedirs(output_dir, exist_ok=True)
        
        output_file = os.path.join(output_dir, "results_summary.csv")
        
        # 기존 파일 읽기
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            # 중복 제거: year, model, sector, ticker, train_test_length, scenarios 기준
            key_columns = ["year", "model", "sector", "ticker", "train_test_length", "scenarios"]
            existing_keys = existing_df[key_columns].apply(tuple, axis=1)
            new_keys = summary_df[key_columns].apply(tuple, axis=1)
            non_duplicate = ~new_keys.isin(existing_keys)
            summary_df = summary_df[non_duplicate]
            # 추가 데이터가 있으면 append
            if not summary_df.empty:
                summary_df.to_csv(output_file, mode='a', header=False, encoding='utf-8-sig', index=False)
                print(f"[INFO] Appended {len(summary_df)} new rows to {output_file}")
            else:
                print(f"[INFO] No new data to append to {output_file}")
        else:
            # 파일이 없으면 새로 생성
            summary_df.to_csv(output_file, encoding='utf-8-sig', index=False)
            print(f"[INFO] Created and saved backtesting summary to {output_file}")

    def BacktestSummary_YOY(self):
        # 기본 경로 설정
        up_down_path = f"./stock_prediction/up_down_signal/{self.args.model_name}/"          
        
        base_input_path = "./stock_prediction/individual_trading/simulation/"
        base_output_path = "./stock_prediction/individual_trading/backtesting/"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(up_down_path) if os.path.isdir(os.path.join(up_down_path, d))]
        
        for year in self.years:
            summary_data = []  # Reset summary_data for each year
            
            for model in tqdm(model_types, desc=f"Processing model types for year {year}"):
                if model == "origin_data":
                    input_path = os.path.join(base_input_path, f'Year_{year}', self.args.model_name, model, time_series_length, self.args.exp_root_path, self.args.sector)
                    scenarios = 0
                else:
                    input_path = os.path.join(base_input_path, f'Year_{year}', self.args.model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                    scenarios = self.args.num_scenarios
                
                if not os.path.exists(input_path):
                    print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                    continue
                
                tickers = [f[:-4] for f in os.listdir(input_path) if f.endswith('.csv')]
                if not tickers:
                    print(f"[WARNING] No tickers found in {input_path}. Skipping.")
                    continue
                
                for ticker in tickers:
                    signal_file = os.path.join(input_path, f"{ticker}.csv")
                    if not os.path.exists(signal_file):
                        print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                        continue
                    
                    df = pd.read_csv(signal_file, index_col=0)
                    
                    df['action'] = df['action'].replace({'No action': 0, 'Buy': 1, 'Sell': -1})
                    
                    no_trade = len(df[df['Margin_Profit'] > 0]) + len(df[df['Margin_Profit'] < 0])
                    
                    max_holding_period = df[df['action'] == 'Holding']['Holding_Period'].max() if no_trade > 0 else 0
                    max_holding_period = 0 if pd.isna(max_holding_period) else max_holding_period
                    
                    mean_holding_period = df[df['action'] == 'Holding']['Holding_Period'].mean() if no_trade > 0 else 0
                    mean_holding_period = 0 if pd.isna(mean_holding_period) else mean_holding_period
                    
                    winning_ratio = len(df[(df['action'] == -1) & (df['Margin_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                    
                    profit_average = df[df['Margin_Profit'] > 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] > 0]) > 0 else 0
                    profit_average = 0 if pd.isna(profit_average) else profit_average
                    loss_average = df[df['Margin_Profit'] < 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] < 0]) > 0 else 0
                    loss_average = 0 if pd.isna(loss_average) else loss_average
                    
                    payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                    loss_sum = df[df['Margin_Profit'] < 0]['Margin_Profit'].sum()
                    profit_sum = df[df['Margin_Profit'] > 0]['Margin_Profit'].sum()
                    profit_factor = -profit_sum / loss_sum if loss_sum < 0 else 0
                    
                    final_cumulative_profit = df['Cumulative_Profit'].iloc[-1]
                    final_cumulative_return = df['Cumulative_Return'].iloc[-1]
                    
                    max_realized_profit = df['Margin_Profit'].max() if no_trade > 0 else 0
                    max_realized_profit = 0 if pd.isna(max_realized_profit) else max_realized_profit
                    max_realized_return = df['Margin_Return'].max() if no_trade > 0 else 0
                    max_realized_return = 0 if pd.isna(max_realized_return) else max_realized_return
                    
                    MDD = df['Drawdown'].min() if no_trade > 0 else 0
                    MDD = 0 if pd.isna(MDD) else MDD
                    MDD_rate = df['Drawdown_rate'].min() if no_trade > 0 else 0
                    MDD_rate = 0 if pd.isna(MDD_rate) else MDD_rate
                    
                    summary_data.append([
                        year, model, self.args.sector, ticker, time_series_length, scenarios,
                        no_trade, max_holding_period, mean_holding_period, winning_ratio,
                        profit_average, loss_average, payoff_ratio, profit_factor,
                        final_cumulative_profit, final_cumulative_return,
                        max_realized_profit, max_realized_return,
                        MDD, MDD_rate
                    ])
            
            summary_df = pd.DataFrame(summary_data, columns=[
                "year", "model", "sector", "ticker", "train_test_length", "scenarios",
                "no_trade", "max_holding_period", "mean_holding_period", "winning_ratio",
                "profit_average", "loss_average", "payoff_ratio", "profit_factor",
                "final_cumulative_profit", "final_cumulative_return",
                "max_realized_profit", "max_realized_return",
                "MaxDrawdown", "MaxDrawdown_rate"
            ])
            
            summary_df = summary_df.round(3)
            
            output_dir = os.path.join(base_output_path, f'Year_{year}')
            os.makedirs(output_dir, exist_ok=True)
            
            output_file = os.path.join(output_dir, "results_summary.csv")
            
            # 기존 파일 읽기
            if os.path.exists(output_file):
                existing_df = pd.read_csv(output_file)
                # 중복 제거: year, model, sector, ticker, train_test_length, scenarios 기준
                key_columns = ["year", "model", "sector", "ticker", "train_test_length", "scenarios"]
                existing_keys = existing_df[key_columns].apply(tuple, axis=1)
                new_keys = summary_df[key_columns].apply(tuple, axis=1)
                non_duplicate = ~new_keys.isin(existing_keys)
                summary_df = summary_df[non_duplicate]
                # 추가 데이터가 있으면 append
                if not summary_df.empty:
                    summary_df.to_csv(output_file, mode='a', header=False, encoding='utf-8-sig', index=False)
                    print(f"[INFO] Appended {len(summary_df)} new rows to {output_file}")
                else:
                    print(f"[INFO] No new data to append to {output_file}")
            else:
                # 파일이 없으면 새로 생성
                summary_df.to_csv(output_file, encoding='utf-8-sig', index=False)
                print(f"[INFO] Created and saved backtesting summary to {output_file}")
                
    def PlotResults(self):
        plt.rcParams.update({
            'axes.titlesize': 40,
            'axes.labelsize': 30,
            'xtick.labelsize': 25,
            'ytick.labelsize': 25,
            'legend.fontsize': 30
        })

        # Define color mapping for models (darker colors)
        colors = {
            'origin_data': '#4A4A4A',  # Darker gray
            'standard_gbm': "#DD53AB",  # Darker violet
            'empirical_gbm': '#FF8C00',  # Darker orange
            'VanillaGAN': "#FFD900",  # Darker yellow
            'WGAN': '#228B22',  # Darker green
            'QuantGAN': "#3D90D4",  # Darker blue
            'DDPM': "#A938E2",  # Darker indigo
            'LDM': "#FA1241",  # Darker red
            'default': '#000000'  # Black (fallback)
        }

        # Define legend label mapping
        legend_labels = {
            'origin_data': 'Origin Data',
            'standard_gbm': 'Standard GBM',
            'empirical_gbm': 'Empirical GBM',
            'VanillaGAN': 'VanillaGAN',
            'WGAN': 'WGAN',
            'QuantGAN': 'QuantGAN',
            'DDPM': 'DDPM',
            'LDM': 'LDM'
        }

        base_input_path = f"./stock_prediction/individual_trading/simulation/Full_period/{self.args.model_name}/"
        base_output_path = "./stock_prediction/individual_trading/plot/"

        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"

        # Get list of model types
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]

        # Get tickers from one of the model directories (assuming all models have the same tickers)
        sample_model = model_types[0] if model_types else None
        if not sample_model:
            print(f"[WARNING] No model types found in {base_input_path}. Skipping.")
            return

        sample_input_path = os.path.join(base_input_path, sample_model, time_series_length, self.args.exp_root_path, self.args.sector)
        if sample_model != "origin_data":
            sample_input_path = os.path.join(base_input_path, sample_model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)

        if not os.path.exists(sample_input_path):
            print(f"[WARNING] Sample input path {sample_input_path} does not exist. Skipping.")
            return

        tickers = [f[:-4] for f in os.listdir(sample_input_path) if f.endswith('.csv')]
        if not tickers:
            print(f"[WARNING] No tickers found in {sample_input_path}. Skipping.")
            return

        for ticker in tqdm(tickers, desc="Processing tickers"):
            # Dictionary to store data for all models
            cumulative_data = {}
            drawdown_data = {}
            buy_and_hold = None

            # Collect data for all models
            for model in model_types:
                method_input_path = os.path.join(base_input_path, model)
                if model == "origin_data":
                    input_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path, self.args.sector)
                else:
                    input_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)

                if not os.path.exists(input_path):
                    print(f"[WARNING] Input path {input_path} does not exist. Skipping model {model}.")
                    continue

                signal_file = os.path.join(input_path, f"{ticker}.csv")
                if not os.path.exists(signal_file):
                    print(f"[WARNING] Signal file {signal_file} does not exist. Skipping model {model}.")
                    continue

                df = pd.read_csv(signal_file, index_col=0)
                df['index'] = pd.to_datetime(df['index'])
                df.set_index('index', inplace=True)

                # Store data for cumulative return and drawdown
                cumulative_data[model] = df['Cumulative_Return']  # Changed from Cumulative_Profit to Cumulative_Return
                drawdown_data[model] = df['Drawdown_rate']

                # Calculate Buy & Hold for the first model (same for all models)
                if buy_and_hold is None:
                    # Buy & Hold 수익률 계산: ((현재 Close - 초기 Close) / 초기 Close) * 100
                    initial_price = df['Close'].iloc[0]
                    df['Investment_Value'] = ((df['Close'] - initial_price) / initial_price) * 100
                    buy_and_hold = df['Investment_Value']

                # Plot individual Trading Signal and Return Size plots
                # 1. Trading Signal Plot
                buy_signals = df[df['action'] == 'Buy']
                sell_signals = df[df['action'] == 'Sell']

                plt.figure(figsize=(24, 16))
                plt.plot(df.index, df['Close'], label='Close Price', color='black', alpha=0.5, linewidth=3)
                plt.scatter(buy_signals.index, buy_signals['Close'], label='Buy', marker='^', color='green', s=250)
                plt.scatter(sell_signals.index, sell_signals['Close'], label='Sell', marker='v', color='red', s=250)
                plt.title(f'Trading signals for {model}', fontsize=40)
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.legend()
                plt.grid(True)

                output_dir = os.path.join(base_output_path, "Trading_signal", model, time_series_length)
                if model != "origin_data":
                    output_dir = os.path.join(output_dir, f"Scenarios_{self.args.num_scenarios}")
                output_dir = os.path.join(output_dir, self.args.exp_root_path, self.args.sector)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ticker}.png")
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
                print(f"[INFO] Saved trading signal plot for {model} to {output_file}")

                # 2. Return Size Plot
                plt.figure(figsize=(24, 16))
                plt.axhline(y=0, color='gray', linestyle='--')
                # Separate positive and negative returns
                positive_returns = df[df['Margin_Return'] >= 0]
                negative_returns = df[df['Margin_Return'] < 0]
                
                # Plot positive returns (green)
                legend_handles = []
                if not positive_returns.empty:
                    plt.scatter(
                        positive_returns.index,
                        positive_returns['Margin_Return'],
                        s=350 * abs(positive_returns['Margin_Return']),
                        alpha=0.5,
                        c='green',
                        label='Positive Realized Return'
                    )
                    legend_handles.append(plt.scatter([], [], s=350, c='green', label='Positive Realized Return'))
                
                # Plot negative returns (red)
                if not negative_returns.empty:
                    plt.scatter(
                        negative_returns.index,
                        negative_returns['Margin_Return'],
                        s=350 * abs(negative_returns['Margin_Return']),
                        alpha=0.5,
                        c='red',
                        label='Negative Realized Return'
                    )
                    legend_handles.append(plt.scatter([], [], s=350, c='red', label='Negative Realized Return'))
                
                # Set Y-axis limits based on max Margin_Return + 5%
                max_return = df['Margin_Return'].max() if not df['Margin_Return'].empty else 0
                min_return = df['Margin_Return'].min() if not df['Margin_Return'].empty else 0
                y_upper = max_return * 1.15 if max_return != 0 else 1  # Add 5% to the maximum return
                y_lower = min_return * 1.20 if min_return < 0 else min_return * 0.95  # Adjust lower bound
                plt.ylim(y_lower, y_upper)
                plt.title(f'Realized return size for {model}', fontsize=40)
                plt.xlabel('Date')
                plt.ylabel('Realized return (%)')
                # Use custom legend handles with fixed scatter size
                if legend_handles:
                    plt.legend(handles=legend_handles)
                else:
                    plt.legend()
                plt.grid(True)

                output_dir = os.path.join(base_output_path, "Return_size", model, time_series_length)
                if model != "origin_data":
                    output_dir = os.path.join(output_dir, f"Scenarios_{self.args.num_scenarios}")
                output_dir = os.path.join(output_dir, self.args.exp_root_path, self.args.sector)
                os.makedirs(output_dir, exist_ok=True)
                output_file = os.path.join(output_dir, f"{ticker}.png")
                plt.tight_layout()
                plt.savefig(output_file)
                plt.close()
                print(f"[INFO] Saved return size plot for {model} to {output_file}")

            # 3. Cumulative Return Comparison Plot
            plt.figure(figsize=(24, 16))
            for model, data in cumulative_data.items():
                plt.plot(data.index, data, label=legend_labels.get(model, model), color=colors.get(model, colors['default']), linewidth=3)
            ''' if buy_and_hold is not None:
                plt.plot(buy_and_hold.index, buy_and_hold, label='Buy & Hold', linestyle='--', color='blue', linewidth=3) '''
            plt.title(f'Cumulative return comparison for {ticker}', fontsize=40)
            plt.xlabel('Date')
            plt.ylabel('Cumulative return (%)')
            plt.legend()
            plt.grid(True)

            output_dir = os.path.join(base_output_path, "Cumulative_return_comparison", time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector,)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{ticker}.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            print(f"[INFO] Saved cumulative return comparison plot to {output_file}")

            # 4. Drawdown Comparison Plot
            plt.figure(figsize=(24, 16))
            for model, data in drawdown_data.items():
                plt.plot(data.index, data, label=legend_labels.get(model, model), color=colors.get(model, colors['default']), linewidth=3)
                plt.fill_between(data.index, 0, data, alpha=0.3, color=colors.get(model, colors['default']), linewidth=3)
            plt.title(f'Drawdown comparison for {ticker}', fontsize=40)
            plt.xlabel('Date')
            plt.ylabel('Drawdown rate (%)')
            plt.legend()
            plt.grid(True)

            output_dir = os.path.join(base_output_path, "Drawdown_comparison", time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
            os.makedirs(output_dir, exist_ok=True)
            output_file = os.path.join(output_dir, f"{ticker}.png")
            plt.tight_layout()
            plt.savefig(output_file)
            plt.close()
            print(f"[INFO] Saved drawdown comparison plot to {output_file}")