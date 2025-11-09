from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.visualization import *
from utils.metric import *

class Portfolio:
    def __init__(self, args):
        self.args = args
        # 현재 작업 디렉토리 출력
        print(f"[INFO] Current working directory: {os.getcwd()}")
        # 메타데이터 파일 경로
        metadata_path = "./datasets/metadata_info.csv"
        print(f"[INFO] Attempting to load metadata from: {metadata_path}")
        # 메타데이터 로드
        try:
            if not os.path.exists(metadata_path):
                raise FileNotFoundError(f"Metadata file {metadata_path} does not exist.")
            self.meta_df = pd.read_csv(metadata_path)
            self.tickers = self.meta_df['Ticker'].tolist() 
            if len(self.tickers) != 150:
                print(f"[WARNING] Expected 150 tickers in metadata, found {len(self.tickers)}")
            else:
                print(f"[INFO] Successfully loaded {len(self.tickers)} tickers from metadata")
        except Exception as e:
            print(f"[ERROR] Failed to load metadata: {str(e)}")
            self.meta_df = None
            self.tickers = []
    
    def UpDown_Signal(self):
        # 기본 경로 설정
        base_input_path = "./stock_prediction/pred_results/GRU_13_64_2/"
        base_output_path = "./stock_prediction/up_down_signal/"
        model_name = "GRU_13_64_2"
                
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 메소드 방식 목록
        methods = ["baseline", "Diffusion", "GAN", "Mathematical"]
        
        for method in methods:
            print(f"[INFO] Processing method: {method}")
            
            # 입력 경로 설정
            method_input_path = os.path.join(base_input_path, method)
            
            # baseline의 경우 세부 모델명 없이 바로 time_series_length 후 exp_root_path로 이동
            if method == "baseline":
                input_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path, self.args.sector)
                # 출력 경로: baseline은 method 폴더 아래 저장, Scenarios 폴더 제외
                output_path = os.path.join(base_output_path, model_name, method, time_series_length, self.args.exp_root_path, self.args.sector)
                subdirs = [""]
            else:
                # Diffusion, GAN, Mathematical은 세부 모델명 폴더 탐색
                subdirs = [d for d in os.listdir(method_input_path) if os.path.isdir(os.path.join(method_input_path, d))]
                if not subdirs:
                    print(f"[WARNING] No submodel directories found for {method}. Skipping.")
                    continue
            
            for subdir in subdirs:
                if method != "baseline":
                    # 입력 경로: 세부 모델명 포함
                    input_path = os.path.join(method_input_path, subdir, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                    # 출력 경로: 알고리즘 방식(method)을 건너뛰고 세부 모델명(subdir)으로 저장
                    output_path = os.path.join(base_output_path, model_name, subdir, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path, self.args.sector)
                
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
    
    def Portfolio_Trading(self):
        # 기본 경로 설정
        base_input_path = "./stock_prediction/up_down_signal/GRU_13_64_2/"
        base_output_path = "./stock_prediction/portfolio_daytrading/portfolio_trading/"
        data_path = f"./datasets/{self.args.exp_root_path}/"  # 섹터별 구분 경로
        model_name = "GRU_13_64_2"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
        
        # 테스트 기간 데이터 길이
        split_num = 753
        
        for model in tqdm(model_types, desc="Processing model types"):
            # 입력 경로 설정
            method_input_path = os.path.join(base_input_path, model)
            
            # baseline의 경우 Scenarios 폴더 없이 바로 exp_root_path
            if model == "baseline":
                input_base_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path)
                output_path = os.path.join(base_output_path, model_name, model, time_series_length, self.args.exp_root_path)
            else:
                input_base_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path)
                output_path = os.path.join(base_output_path, model_name, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path)
            
            if not os.path.exists(input_base_path):
                print(f"[WARNING] Input base path {input_base_path} does not exist. Skipping.")
                continue
            
            # 메타데이터에서 종목 리스트 확인
            if not self.tickers:
                print(f"[ERROR] No tickers available. Metadata load may have failed.")
                continue
            tickers = self.tickers
            if len(tickers) != 150:
                print(f"[WARNING] Expected 150 tickers, found {len(tickers)}. Skipping.")
                continue
            
            # 모든 종목의 Predicted 값을 수집
            signal_data_list = []
            date_index = None
            missing_tickers = []
            
            for ticker in tickers:
                # 티커의 섹터 확인
                ticker_sector = self.meta_df[self.meta_df['Ticker'] == ticker]['Sector'].iloc[0]
                input_path = os.path.join(input_base_path, ticker_sector)
                
                # 신호 데이터 로드
                signal_file = os.path.join(input_path, f"{ticker}.csv")
                if not os.path.exists(signal_file):
                    print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                    missing_tickers.append(ticker)
                    continue
                
                signal_data = pd.read_csv(signal_file).reset_index(drop=True)
                signal_data = signal_data[['Predicted']].rename(columns={'Predicted': ticker})
                signal_data_list.append(signal_data)
                
                # 날짜 인덱스 설정 (첫 번째 종목에서 가져옴)
                if date_index is None:
                    stock_file = os.path.join(data_path, ticker_sector, f"{ticker}.csv")
                    if os.path.exists(stock_file):
                        stock_data = pd.read_csv(stock_file).iloc[-split_num:].reset_index(drop=True)
                        date_index = stock_data['index'] if 'index' in stock_data.columns else pd.RangeIndex(len(stock_data))
            
            if not signal_data_list:
                print(f"[WARNING] No valid signal data for model {model}. Skipping.")
                continue
            
            # 모든 티커 데이터와 Date를 한 번에 병합
            signal_df = pd.DataFrame({'Date': date_index})
            signal_df = pd.concat([signal_df] + signal_data_list, axis=1)
            
            # 누락된 티커 처리: 0으로 채우기
            for ticker in missing_tickers:
                signal_df[ticker] = 0
            
            # 컬럼 순서 정렬
            signal_df = signal_df[['Date'] + tickers]
            
            # 출력 저장
            output_file = os.path.join(output_path, "signal_data.csv")
            os.makedirs(output_path, exist_ok=True)
            signal_df.to_csv(output_file, index=False)
            print(f"[INFO] Saved signal data for model {model} to {output_file}")
            
    def Portfolio_Construction(self):
        # 기본 경로 설정
        base_input_path = "./stock_prediction/portfolio_daytrading/portfolio_trading/GRU_13_64_2/"
        base_output_path = "./stock_prediction/portfolio_daytrading/portfolio_results/GRU_13_64_2/"
        data_path = f"./datasets/{self.args.exp_root_path}/"
        model_name = "GRU_13_64_2"
        
        # 시계열 길이 폴더 이름
        time_series_length = f"Train_{self.args.train_months}_Test_{self.args.sliding_test_months}"
        
        # 모델 타입 동적 읽기
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
        
        # 거래 설정
        initial_seed = 100000  # 초기 시드 금액
        commission_rate = 0.0005  # 수수료 0.05%
        split_num = 753  # 테스트 기간 데이터 길이
        
        for model in tqdm(model_types, desc="Processing model types"):
            # 입력/출력 경로 설정
            method_input_path = os.path.join(base_input_path, model)
            if model == "baseline":
                input_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path)
                output_base_path = os.path.join(base_output_path, model, time_series_length, self.args.exp_root_path)
            else:
                input_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path)
                output_base_path = os.path.join(base_output_path, model, time_series_length, f"Scenarios_{self.args.num_scenarios}", self.args.exp_root_path)
            
            if not os.path.exists(input_path):
                print(f"[WARNING] Input path {input_path} does not exist. Skipping.")
                continue
            
            # 신호 데이터 로드
            signal_file = os.path.join(input_path, "signal_data.csv")
            if not os.path.exists(signal_file):
                print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                continue
            signal_df = pd.read_csv(signal_file)
            
            # 포트폴리오 유형 정의
            portfolio_types = {
                'all_tickers': {'key': 'all', 'tickers': self.tickers},
                'sector': {'key': 'Sector', 'values': self.meta_df['Sector'].unique()},
                'risk_category': {'key': 'Risk_Category', 'values': self.meta_df['Risk_Category'].unique()},
                'market_cap': {'key': 'Market Cap Classification', 'values': self.meta_df['Market Cap Classification'].unique()}
            }
            
            for portfolio_type, config in portfolio_types.items():
                if portfolio_type == 'all_tickers':
                    # 전체 종목 포트폴리오
                    portfolio_keys = [config['key']]
                else:
                    # 섹터별, 투자 성향별, 시총 규모별
                    portfolio_keys = config['values']
                
                for portfolio_key in portfolio_keys:
                    # 포트폴리오 티커 선별
                    if portfolio_type == 'all_tickers':
                        selected_tickers = self.tickers
                    else:
                        selected_tickers = self.meta_df[self.meta_df[config['key']] == portfolio_key]['Ticker'].tolist()
                    
                    # 포트폴리오 거래 결과 저장용 데이터프레임
                    portfolio_data = {
                        'index': [],  # 날짜
                        'Selected_Tickers': [],
                        'Margin_Profit': [],
                        'Margin_Return': [],
                        'Cumulative_Profit': [],
                        'Cumulative_Return': [],
                        'Drawdown': [],
                        'Drawdown_rate': []
                    }
                    
                    cumulative_profit = 0
                    peak_profit = 0
                    peak_return = 0
                    
                    # 각 날짜 처리
                    for i in range(len(signal_df) - 1):  # 마지막 날은 매수만 가능하므로 제외
                        daily_signals = signal_df.iloc[i]
                        daily_selected_tickers = [ticker for ticker in selected_tickers if daily_signals[ticker] == 1]
                        
                        if not daily_selected_tickers:
                            # 상승 예측 종목 없음
                            portfolio_data['index'].append(daily_signals['Date'])
                            portfolio_data['Selected_Tickers'].append("")
                            portfolio_data['Margin_Profit'].append(0)
                            portfolio_data['Margin_Return'].append(0)
                            portfolio_data['Cumulative_Profit'].append(cumulative_profit)
                            portfolio_data['Cumulative_Return'].append((cumulative_profit / initial_seed) * 100 if initial_seed != 0 else 0)
                            portfolio_data['Drawdown'].append(-(peak_profit - cumulative_profit))
                            portfolio_data['Drawdown_rate'].append(-(peak_return - (cumulative_profit / initial_seed * 100)))
                            continue
                        
                        # 주식 가격 데이터 로드
                        daily_return = 0
                        for ticker in daily_selected_tickers:
                            ticker_sector = self.meta_df[self.meta_df['Ticker'] == ticker]['Sector'].iloc[0]
                            stock_file = os.path.join(data_path, ticker_sector, f"{ticker}.csv")
                            if not os.path.exists(stock_file):
                                print(f"[WARNING] Stock file {stock_file} does not exist. Skipping.")
                                continue
                            
                            stock_data = pd.read_csv(stock_file).iloc[-split_num:].reset_index(drop=True)
                            if i + 1 >= len(stock_data):
                                continue
                            
                            open_price = stock_data.iloc[i + 1]['Open']
                            close_price = stock_data.iloc[i + 1]['Close']
                            
                            if pd.isna(open_price) or pd.isna(close_price):
                                print(f"[WARNING] Missing price data for {ticker} on day {i+1}. Skipping.")
                                continue
                            
                            # 균등 가중치
                            weight = 1.0 / len(daily_selected_tickers)
                            # 수익률: (종가 - 시가) / 시가 - 수수료
                            return_ = ((close_price - open_price) / open_price - commission_rate) * weight
                            daily_return += return_
                        
                        # 실현 수익
                        margin_profit = initial_seed * daily_return
                        cumulative_profit += margin_profit
                        cumulative_return = (cumulative_profit / initial_seed) * 100 if initial_seed != 0 else 0
                        
                        # 드로다운 계산
                        if cumulative_profit > peak_profit:
                            peak_profit = cumulative_profit
                        drawdown = -(peak_profit - cumulative_profit)
                        
                        if cumulative_return > peak_return:
                            peak_return = cumulative_return
                        drawdown_rate = -(peak_return - cumulative_return)
                        
                        # 결과 저장
                        portfolio_data['index'].append(daily_signals['Date'])
                        portfolio_data['Selected_Tickers'].append(','.join(daily_selected_tickers))
                        portfolio_data['Margin_Profit'].append(margin_profit)
                        portfolio_data['Margin_Return'].append(daily_return * 100)
                        portfolio_data['Cumulative_Profit'].append(cumulative_profit)
                        portfolio_data['Cumulative_Return'].append(cumulative_return)
                        portfolio_data['Drawdown'].append(drawdown)
                        portfolio_data['Drawdown_rate'].append(drawdown_rate)
                    
                    # 데이터프레임 생성
                    portfolio_df = pd.DataFrame(portfolio_data)
                    portfolio_df = portfolio_df.round(3)
                    
                    # 출력 저장
                    output_dir = os.path.join(output_base_path, portfolio_type)
                    output_file = os.path.join(output_dir, f"{portfolio_key.replace(' ', '_')}.csv")
                    os.makedirs(output_dir, exist_ok=True)
                    portfolio_df.to_csv(output_file, index=False)
                    print(f"[INFO] Saved portfolio results for {portfolio_type}/{portfolio_key} to {output_file}")
                    
    def Portfolio_Summary(self):
        base_input_path = "./stock_prediction/portfolio_daytrading/portfolio_results/GRU_13_64_2/"
        base_output_path = "./stock_prediction/portfolio_daytrading/portfolio_backtesting/"
        model_name = "GRU_13_64_2"
        
        # 조합식 정의
        TRAIN_MONTHS_LIST = [12, 24, 36]
        NUM_SCENARIOS_LIST = [25, 50, 75, 100]
        model_types = [d for d in os.listdir(base_input_path) if os.path.isdir(os.path.join(base_input_path, d))]
        portfolio_types = ['all_tickers', 'sector', 'risk_category', 'market_cap']
        
        summary_data = []
        
        # 모든 조합식 순회
        for train_months in tqdm(TRAIN_MONTHS_LIST, desc="Processing train months"):
            for num_scenarios in NUM_SCENARIOS_LIST:
                time_series_length = f"Train_{train_months}_Test_{self.args.sliding_test_months}"
                
                for model in model_types:
                    method_input_path = os.path.join(base_input_path, model)
                    if model == "baseline":
                        input_base_path = os.path.join(method_input_path, time_series_length, self.args.exp_root_path)
                        scenarios = 0
                    else:
                        input_base_path = os.path.join(method_input_path, time_series_length, f"Scenarios_{num_scenarios}", self.args.exp_root_path)
                        scenarios = num_scenarios
                    
                    if not os.path.exists(input_base_path):
                        print(f"[WARNING] Input path {input_base_path} does not exist. Skipping.")
                        continue
                    
                    for portfolio_type in portfolio_types:
                        portfolio_path = os.path.join(input_base_path, portfolio_type)
                        if not os.path.exists(portfolio_path):
                            print(f"[WARNING] Portfolio path {portfolio_path} does not exist. Skipping.")
                            continue
                        
                        portfolio_keys = [f[:-4] for f in os.listdir(portfolio_path) if f.endswith('.csv')]
                        if not portfolio_keys:
                            print(f"[WARNING] No portfolio files found in {portfolio_path}. Skipping.")
                            continue
                        
                        for portfolio_key in portfolio_keys:
                            signal_file = os.path.join(portfolio_path, f"{portfolio_key}.csv")
                            if not os.path.exists(signal_file):
                                print(f"[WARNING] Signal file {signal_file} does not exist. Skipping.")
                                continue
                            
                            df = pd.read_csv(signal_file)
                            
                            # 거래 횟수 (Margin_Profit이 0이 아닌 경우)
                            no_trade = len(df[df['Margin_Profit'] != 0])
                            
                            # 보유 기간 계산
                            df['Holding_Period'] = (df['Selected_Tickers'] != '').astype(int).cumsum()
                            max_holding_period = df[df['Selected_Tickers'] != '']['Holding_Period'].max() if no_trade > 0 else 0
                            max_holding_period = 0 if pd.isna(max_holding_period) else max_holding_period
                            mean_holding_period = df[df['Selected_Tickers'] != '']['Holding_Period'].mean() if no_trade > 0 else 0
                            mean_holding_period = 0 if pd.isna(mean_holding_period) else mean_holding_period
                            
                            # 승률 (Margin_Profit > 0인 경우)
                            winning_ratio = len(df[(df['Margin_Profit'] > 0)]) / no_trade if no_trade > 0 else 0
                            
                            # 평균 수익/손실
                            profit_average = df[df['Margin_Profit'] > 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] > 0]) > 0 else 0
                            profit_average = 0 if pd.isna(profit_average) else profit_average
                            loss_average = df[df['Margin_Profit'] < 0]['Margin_Profit'].mean() if len(df[df['Margin_Profit'] < 0]) > 0 else 0
                            loss_average = 0 if pd.isna(loss_average) else loss_average
                            
                            # 페이오프 비율
                            payoff_ratio = profit_average / -loss_average if loss_average < 0 else 0
                            
                            # 수익률 요인
                            loss_sum = df[df['Margin_Profit'] < 0]['Margin_Profit'].sum()
                            profit_sum = df[df['Margin_Profit'] > 0]['Margin_Profit'].sum()
                            profit_factor = -profit_sum / loss_sum if loss_sum < 0 else 0
                            
                            # 최종 누적 수익 및 수익률
                            final_cumulative_profit = df['Cumulative_Profit'].iloc[-1]
                            final_cumulative_return = df['Cumulative_Return'].iloc[-1]
                            
                            # 최대 실현 수익/수익률
                            max_realized_profit = df['Margin_Profit'].max() if no_trade > 0 else 0
                            max_realized_profit = 0 if pd.isna(max_realized_profit) else max_realized_profit
                            max_realized_return = df['Margin_Return'].max() if no_trade > 0 else 0
                            max_realized_return = 0 if pd.isna(max_realized_return) else max_realized_return
                            
                            # 최대 손실 (MDD)
                            MDD = df['Drawdown'].min() if no_trade > 0 else 0
                            MDD = 0 if pd.isna(MDD) else MDD
                            MDD_rate = df['Drawdown_rate'].min() if no_trade > 0 else 0
                            MDD_rate = 0 if pd.isna(MDD_rate) else MDD_rate
                            
                            summary_data.append([
                                model, portfolio_type, portfolio_key, time_series_length, scenarios,
                                no_trade, max_holding_period, mean_holding_period, winning_ratio,
                                profit_average, loss_average, payoff_ratio, profit_factor,
                                final_cumulative_profit, final_cumulative_return,
                                max_realized_profit, max_realized_return,
                                MDD, MDD_rate
                            ])
        
        summary_df = pd.DataFrame(summary_data, columns=[
            "model", "portfolio_type", "portfolio_key", "train_test_length", "scenarios",
            "no_trade", "max_holding_period", "mean_holding_period", "winning_ratio",
            "profit_average", "loss_average", "payoff_ratio", "profit_factor",
            "final_cumulative_profit", "final_cumulative_return",
            "max_realized_profit", "max_realized_return",
            "MaxDrawdown", "MaxDrawdown_rate"
        ])
        
        summary_df = summary_df.round(3)
        
        output_dir = os.path.join(base_output_path, "full_period")
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "results_summary.csv")
        
        if os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            key_columns = ["model", "portfolio_type", "portfolio_key", "train_test_length", "scenarios"]
            existing_keys = existing_df[key_columns].apply(tuple, axis=1)
            new_keys = summary_df[key_columns].apply(tuple, axis=1)
            non_duplicate = ~new_keys.isin(existing_keys)
            summary_df = summary_df[non_duplicate]
            if not summary_df.empty:
                summary_df.to_csv(output_file, mode='a', header=False, encoding='utf-8-sig', index=False)
                print(f"[INFO] Appended {len(summary_df)} new rows to {output_file}")
            else:
                print(f"[INFO] No new data to append to {output_file}")
        else:
            summary_df.to_csv(output_file, encoding='utf-8-sig', index=False)
            print(f"[INFO] Created and saved portfolio summary to {output_file}")