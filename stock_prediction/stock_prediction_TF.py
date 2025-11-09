from common_imports import *
from utils.sliding_window import get_sliding_window_data
from utils.feature_preprocessing import calculate_indicators, add_labels

plt.rcParams.update({
    'axes.titlesize': 25,    # 제목 폰트 크기
    'axes.labelsize': 20,    # Y축 레이블 폰트 크기 (metric 강조)
    'xtick.labelsize': 15,   # X축 눈금 레이블 폰트 크기
    'ytick.labelsize': 15,   # Y축 눈금 레이블 폰트 크기
    'legend.fontsize': 20    # 범례 폰트 크기
})

# 데이터셋 클래스
class StockScenarioDataset(Dataset):
    def __init__(self, data_df, seq_len=127, mode='train'):
        self.data_df = data_df
        self.seq_len = seq_len
        self.mode = mode
        self.feature_cols = [
            'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist',
            'MOM', 'PPO', 'ROC', 'ROCR', 'RSI',
            'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX'
        ]
        self.label_col = 'Up_Down'

        self.data_df[self.feature_cols] = self.data_df[self.feature_cols].fillna(0).replace([np.inf, -np.inf], 0)
        scaler = StandardScaler()
        self.data_df[self.feature_cols] = scaler.fit_transform(self.data_df[self.feature_cols])

        if mode == 'train':
            self.scenario_groups = self.data_df.groupby('scenario_id')
            self.scenario_ids = list(self.scenario_groups.groups.keys())
        else:
            self.scenario_groups = None
            self.scenario_ids = ['test']

        self.valid_indices = []
        self.scenario_mapping = []
        if mode == 'train':
            for scenario_id in self.scenario_ids:
                scenario_data = self.scenario_groups.get_group(scenario_id)
                for i in range(len(scenario_data) - self.seq_len):
                    self.valid_indices.append((scenario_id, i))
                    self.scenario_mapping.append(scenario_id)
        else:
            for i in range(len(self.data_df) - self.seq_len):
                self.valid_indices.append(('test', i))
                self.scenario_mapping.append('test')

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        scenario_id, start_idx = self.valid_indices[idx]
        if self.mode == 'train':
            scenario_data = self.scenario_groups.get_group(scenario_id)
        else:
            scenario_data = self.data_df

        seq_data = scenario_data[self.feature_cols].iloc[start_idx:start_idx + self.seq_len].values
        label = scenario_data[self.label_col].iloc[start_idx + self.seq_len]

        seq_data = torch.tensor(seq_data, dtype=torch.float)
        label = torch.tensor(label, dtype=torch.float)

        return seq_data, label, scenario_id

# Transformer 모델
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class StockTransformer(nn.Module):
    def __init__(self, input_dim=13, d_model=64, nhead=4, num_layers=3, dropout=0.1, seq_len=127):
        super(StockTransformer, self).__init__()
        self.input_dim = input_dim
        self.d_model = d_model
        self.seq_len = seq_len
        self.nhead = nhead
        self.num_layers = num_layers

        self.input_linear = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, seq_len)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        self.fc = nn.Linear(d_model, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        x = self.input_linear(x)
        residual = x  # 잔차 연결을 위한 입력 저장
        x = self.pos_encoder(x)
        
        if return_attention:
            attn_weights = []
            for layer in self.transformer_encoder.layers:
                x, attn = layer.self_attn(x, x, x, need_weights=True)
                attn_weights.append(attn)
            attn_weights = attn_weights[-1]  # [batch, seq_len, seq_len]
            x = self.transformer_encoder(x)
        else:
            x = self.transformer_encoder(x)

        x = x + residual  # 잔차 연결
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.fc(x)
        if return_attention:
            return x, attn_weights
        return x

    def get_model_name(self):
        return f"Transformer_{self.input_dim}_{self.d_model}_{self.num_layers}"

# 학습 함수
def train_model(model, train_loader, criterion, optimizer, num_epochs, device):
    model.train()
    scaler = GradScaler()
    progress = tqdm(range(num_epochs))

    for epoch in progress:
        running_loss = 0.0
        all_labels = []
        all_preds = []
        all_probs = []  # Store probabilities for mean calculation
        total = 0
        correct = 0

        for data, labels, _ in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = model(data).squeeze(1)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            probabilities = torch.sigmoid(outputs)
            all_probs.extend(probabilities.detach().cpu().numpy())  # Detach before converting to numpy

            all_labels.extend(labels.cpu().numpy())
            total += labels.size(0)

        # Calculate mean probability as threshold
        mean_prob = np.mean(all_probs) if all_probs else 0.5  # Fallback to 0.5 if no probs
        # Convert probabilities to predictions using mean_prob
        predicted = (np.array(all_probs) > mean_prob).astype(float)
        all_preds.extend(predicted)
        correct = (predicted == np.array(all_labels)).sum()

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        precision = precision_score(all_labels, all_preds, zero_division=0)
        recall = recall_score(all_labels, all_preds, zero_division=0)
        f1 = f1_score(all_labels, all_preds, zero_division=0)

        msg = f"Epoch [{epoch+1}/{num_epochs}] Loss: {train_loss:.4f} | Accuracy: {train_acc:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1-Score: {f1:.4f} | Mean Prob: {mean_prob:.4f}"
        progress.set_description(msg)

    torch.cuda.empty_cache()

# 테스트 함수
def test_model(model, test_loader, criterion, device, scenario_ids, output_dir, window_idx, train_dataset, test_df):
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0
    results = []
    attention_contributions = []
    all_probs = []  # Store probabilities for mean calculation
    all_labels = []  # Store labels for accuracy calculation

    # 학습 데이터에서 시나리오별 대표 피처 추출
    scenario_features = {}
    for scenario_id in scenario_ids:
        scenario_data = train_dataset.data_df[train_dataset.data_df['scenario_id'] == scenario_id][train_dataset.feature_cols].values
        if len(scenario_data) > 0:
            scenario_features[scenario_id] = np.mean(scenario_data, axis=0)

    # 테스트 데이터의 가격 경로 추출 및 정규화
    prices = test_df['Adj Close'].values
    if len(prices) > 0:
        prices_normalized = (prices - prices.min()) / (prices.max() - prices.min() + 1e-8)
    else:
        prices_normalized = np.zeros(len(test_df))

    with torch.no_grad():
        for data, labels, scenario_ids_batch in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs, attn_weights = model(data, return_attention=True)
            outputs = outputs.squeeze(1)
            loss = criterion(outputs, labels)
            test_loss += loss.item()

            probabilities = torch.sigmoid(outputs)
            all_probs.extend(probabilities.detach().cpu().numpy())  # Detach for safety
            all_labels.extend(labels.cpu().numpy())  # Collect labels
            total += labels.size(0)

            # attn_weights shape: [batch, nhead, seq_len, seq_len]
            attn_weights = attn_weights.mean(dim=1)  # [batch, seq_len, seq_len]

            for i, scenario_id in enumerate(scenario_ids_batch):
                contribution = {sid: 0.0 for sid in scenario_ids}
                if scenario_id == 'test':
                    test_features = data[i].cpu().numpy().mean(axis=0)
                    similarities = {}
                    for sid, features in scenario_features.items():
                        sim = cosine_similarity([test_features], [features])[0][0]
                        similarities[sid] = max(sim, 0.0)
                    for sid in scenario_ids:
                        contribution[sid] = similarities.get(sid, 0.0)
                else:
                    contribution[scenario_id] = attn_weights[i].mean().item()

                attention_contributions.append({
                    'date': len(attention_contributions),
                    'contributions': contribution
                })

            results.extend(zip(labels.cpu().numpy(), probabilities.cpu().numpy()))

    # Calculate mean probability as threshold
    mean_prob = np.mean(all_probs) if all_probs else 0.5  # Fallback to 0.5 if no probs
    # Convert probabilities to predictions using mean_prob
    predicted = (np.array(all_probs) > mean_prob).astype(float)
    correct = (predicted == np.array(all_labels)).sum()

    test_loss = test_loss / len(test_loader)
    test_acc = 100 * correct / total
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%, Mean Prob: {mean_prob:.4f}")

    contribution_df = pd.DataFrame([
        {**{'date': entry['date']}, **entry['contributions']}
        for entry in attention_contributions
    ])
    contribution_df = contribution_df.groupby('date').sum()

    # 어텐션 기여도 저장
    attention_output_dir = output_dir.replace('pred_results', 'attention_results')
    os.makedirs(attention_output_dir, exist_ok=True)
    contribution_df.to_csv(f'{attention_output_dir}/window_{window_idx}_raw_attention_contributions.csv')

    # 시각화: 히트맵 스타일
    plt.figure(figsize=(30, 15))
    df_pivot = contribution_df.transpose()
    sns.heatmap(df_pivot, cmap='inferno', cbar=True, xticklabels=20, yticklabels=False)

    plt.title(f'Scenario activation rate - (Window {window_idx})')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('')
    plt.xticks(ha='right')
    plt.xlim(0, len(contribution_df))
    plt.tight_layout()
    plt.savefig(f'{attention_output_dir}/window_{window_idx}_raw_attention_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    # 시각화: 개별 fill_between 스타일
    fig, ax1 = plt.subplots(figsize=(30, 15))
    cmap = plt.cm.inferno
    num_scenarios = len(contribution_df.columns)
    colors = [cmap(i / num_scenarios) for i in range(num_scenarios)]

    max_contribution = contribution_df.max().max()
    y_limit = max_contribution * 1.2 if max_contribution > 0 else 1.0

    for idx, column in enumerate(contribution_df.columns):
        contribution_values = contribution_df[column].values
        ax1.fill_between(
            contribution_df.index,
            0,
            contribution_values,
            color=colors[idx],
            alpha=0.2
        )
        ax1.plot(
            contribution_df.index,
            contribution_values,
            color=colors[idx],
            linewidth=1.5,
            alpha=0.6
        )

    ax1.set_xlabel('Index')
    ax1.set_ylabel('Activation rate')
    ax1.set_title(f'Scenario activation rate - (Window {window_idx})')
    ax1.set_ylim(0, y_limit)
    ax1.set_xlim(0, len(contribution_df))
    ax1.tick_params(axis='y')
    ax1.tick_params(axis='x')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        contribution_df.index,
        prices_normalized,
        label='Normalized Price',
        color='black',
        linewidth=2.5
    )
    ax2.set_ylabel('Normalized Price')
    ax2.set_ylim(0, 1)
    ax2.tick_params(axis='y')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(f'{attention_output_dir}/window_{window_idx}_raw_attention_filled_plot.png', dpi=300, bbox_inches='tight')
    plt.close()

    return results, contribution_df


# 모델 저장 함수
def save_model(model, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

class Exp_STP_TF:
    def __init__(self, args):
        self.args = args

    def select_balanced_scenarios(self, scenario_df, num_scenarios):
        scenario_columns = scenario_df.columns

        scenario_stats = []
        for scenario_id in scenario_columns:
            prices = scenario_df[scenario_id].values
            log_returns = np.diff(np.log(prices))
            drift = np.mean(log_returns) * 252
            volatility = np.std(log_returns) * np.sqrt(252)
            scenario_stats.append({'scenario_id': scenario_id, 'Drift': drift, 'Volatility': volatility})

        stats_df = pd.DataFrame(scenario_stats)

        drift_quantiles = stats_df['Drift'].quantile([0.333, 0.667]).values
        volatility_quantiles = stats_df['Volatility'].quantile([0.333, 0.667]).values

        def classify_risk(drift, volatility):
            drift_level = 'Low' if drift <= drift_quantiles[0] else 'Medium' if drift <= drift_quantiles[1] else 'High'
            vol_level = 'Low' if volatility <= volatility_quantiles[0] else 'Medium' if volatility <= volatility_quantiles[1] else 'High'
            if vol_level == 'High' and drift_level == 'Low':
                return 'High Risk'
            elif vol_level == 'Low' and drift_level in ['Medium', 'High']:
                return 'Low Risk'
            else:
                return 'Medium Risk'

        stats_df['Risk'] = stats_df.apply(lambda row: classify_risk(row['Drift'], row['Volatility']), axis=1)

        risk_groups = stats_df.groupby('Risk')
        selected_scenarios = []

        low_risk_target = int(num_scenarios * 0.25)
        medium_risk_target = int(num_scenarios * 0.50)
        high_risk_target = int(num_scenarios * 0.25)

        for risk, target_count in [('Low Risk', low_risk_target), ('Medium Risk', medium_risk_target), ('High Risk', high_risk_target)]:
            if risk in risk_groups.groups:
                group = risk_groups.get_group(risk)
                sample_count = min(len(group), target_count)
                if sample_count > 0:
                    selected = group.sample(n=sample_count, replace=False, random_state=42)
                    selected_scenarios.extend(selected['scenario_id'].tolist())
            else:
                print(f"[WARNING] {risk} 그룹에 시나리오가 없습니다.")

        if len(selected_scenarios) < num_scenarios:
            shortfall = num_scenarios - len(selected_scenarios)
            remaining_scenarios = stats_df[~stats_df['scenario_id'].isin(selected_scenarios)]
            if len(remaining_scenarios) > 0:
                additional = remaining_scenarios.sample(n=min(shortfall, len(remaining_scenarios)), replace=False, random_state=42)
                selected_scenarios.extend(additional['scenario_id'].tolist())
            else:
                print(f"[WARNING] 추가로 선택할 수 있는 시나리오가 없습니다. (현재 선택된 시나리오 수: {len(selected_scenarios)}/{num_scenarios})")

        return selected_scenarios, stats_df, drift_quantiles, volatility_quantiles

    def prepare_scenario_data(self, train_df, scenario_df, seq_len, num_scenarios):
        train_df = train_df.copy()
        train_df['index'] = pd.to_datetime(train_df['index'])
        scenario_df = scenario_df.copy()

        all_scenarios = []
        selected_scenarios, _, _, _ = self.select_balanced_scenarios(scenario_df, num_scenarios)

        for scenario_id in selected_scenarios:
            temp_df = train_df.copy()
            scenario_series = pd.Series(scenario_df[scenario_id].values, name='Adj Close')
            scenario_temp_df = pd.DataFrame({
                'index': pd.date_range(start=train_df['index'].iloc[-1] + pd.Timedelta(days=1),
                                       periods=len(scenario_series), freq='D'),
                'Adj Close': scenario_series
            })

            combined_df = pd.concat([temp_df, scenario_temp_df], ignore_index=True)
            combined_df = combined_df.sort_values('index').reset_index(drop=True)
            combined_df = calculate_indicators(combined_df)
            combined_df = add_labels(combined_df)

            indicator_columns = [
                'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist',
                'MOM', 'PPO', 'ROC', 'ROCR', 'RSI',
                'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'Up_Down'
            ]

            scenario_start_idx = len(train_df)
            processed_scenario_df = combined_df.iloc[scenario_start_idx - seq_len:].reset_index(drop=True)
            processed_scenario_df = processed_scenario_df[indicator_columns]
            processed_scenario_df = processed_scenario_df.dropna().reset_index(drop=True)
            processed_scenario_df['scenario_id'] = scenario_id
            all_scenarios.append(processed_scenario_df)

        processed_train_df = pd.concat(all_scenarios, ignore_index=True)
        return processed_train_df

    def prepare_test_data(self, train_df, test_df, seq_len):
        train_df = train_df.copy()
        test_df = test_df.copy()
        train_df['index'] = pd.to_datetime(train_df['index'])
        test_df['index'] = pd.to_datetime(test_df['index'])

        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        combined_df = combined_df.sort_values('index').reset_index(drop=True)
        combined_df = calculate_indicators(combined_df)
        combined_df = add_labels(combined_df)

        test_start_idx = len(train_df)
        processed_test_df = combined_df.iloc[test_start_idx - seq_len:].reset_index(drop=True)

        indicator_columns = [
            'APO', 'CMO', 'MACD', 'MACD_signal', 'MACD_hist',
            'MOM', 'PPO', 'ROC', 'ROCR', 'RSI',
            'STOCHRSI_fastk', 'STOCHRSI_fastd', 'TRIX', 'Up_Down'
        ]
        processed_test_df = processed_test_df[indicator_columns]
        processed_test_df = processed_test_df.dropna().reset_index(drop=True)

        return processed_test_df, combined_df.iloc[test_start_idx:].reset_index(drop=True)  # 원본 테스트 데이터 반환 추가

    def process(self):
        base_path = f'./outputs/{self.args.GBM_type}'
        stock_files = [f for f in os.listdir(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector)) if f.endswith('.csv')]
        
        for stock_file in stock_files:
            ticker_name = stock_file[:-4]
            print(f"\n[INFO] Simulating for GBM type : {self.args.GBM_type} | sector : {self.args.sector} | ticker: {ticker_name}")

            stock_data = pd.read_csv(os.path.join('./datasets/', self.args.exp_root_path, self.args.sector, stock_file))
            sliding_windows = get_sliding_window_data(
                stock_data,
                date_col='index',
                test_start_year=self.args.test_start_year,
                total_test_months=self.args.total_test_months,
                sliding_test_months=self.args.sliding_test_months,
                train_months=self.args.train_months
            )

            for window_idx, (train_df, test_df, window_info) in enumerate(sliding_windows):
                scenario_df = pd.read_csv(os.path.join(
                    base_path,
                    f'Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/',
                    f'GBM_scenario_2/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/',
                    f"Sliding_Window_{window_idx+1}_simulated_paths.csv"
                ))

                selected_scenarios, stats_df, drift_quantiles, volatility_quantiles = self.select_balanced_scenarios(
                    scenario_df, self.args.num_scenarios
                )

                plt.figure(figsize=(20, 15))
                colors = {'Low Risk': 'green', 'Medium Risk': 'orange', 'High Risk': 'red'}
                for category in stats_df['Risk'].unique():
                    subset = stats_df[stats_df['Risk'] == category]
                    plt.scatter(subset['Drift'], subset['Volatility'],
                                c=colors[category], label=category, alpha=0.6, s=200)

                selected_subset = stats_df[stats_df['scenario_id'].isin(selected_scenarios)]
                for category in selected_subset['Risk'].unique():
                    subset = selected_subset[selected_subset['Risk'] == category]
                    plt.scatter(subset['Drift'], subset['Volatility'],
                                c=colors[category], s=400, edgecolors='black', linewidths=2, alpha=0.8,
                                label=f'Selected {category}' if category in stats_df['Risk'].unique() else None)

                plt.axvline(x=drift_quantiles[0], color='gray', linestyle='--', alpha=0.5)
                plt.axvline(x=drift_quantiles[1], color='gray', linestyle='--', alpha=0.5)
                plt.axhline(y=volatility_quantiles[0], color='gray', linestyle='--', alpha=0.5)
                plt.axhline(y=volatility_quantiles[1], color='gray', linestyle='--', alpha=0.5)

                plt.xlabel('Drift')
                plt.ylabel('Volatility')
                plt.title(f'Scenario Risk Classification for {self.args.sector} - {ticker_name} (Window {window_idx+1})')
                plt.legend()
                plt.grid(True, alpha=0.3)

                save_path = f'./stock_prediction/risk_plots/{self.args.GBM_type}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/Scenarios_{self.args.num_scenarios}/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/'
                os.makedirs(save_path, exist_ok=True)
                plt.tight_layout()
                plt.savefig(f'{save_path}window_{window_idx+1}_risk_classification.png', bbox_inches='tight')
                plt.close()

                processed_train_df = self.prepare_scenario_data(
                    train_df, scenario_df, seq_len=self.args.seq_len, num_scenarios=self.args.num_scenarios
                )
                processed_test_df, test_df_original = self.prepare_test_data(
                    train_df, test_df, seq_len=self.args.seq_len
                )

                train_dataset = StockScenarioDataset(processed_train_df, seq_len=self.args.seq_len, mode='train')
                test_dataset = StockScenarioDataset(processed_test_df, seq_len=self.args.seq_len, mode='test')
                train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=True)
                test_loader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False)

                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                model = StockTransformer(
                    input_dim=len(train_dataset.feature_cols),
                    seq_len=self.args.seq_len
                ).to(device)
                model_name = model.get_model_name()
                criterion = nn.BCEWithLogitsLoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

                print(f"[INFO] Training model for Scenarios {self.args.num_scenarios} | {self.args.sector} | {ticker_name} | Window {window_idx+1}")
                train_model(model, train_loader, criterion, optimizer, self.args.num_epochs, device)

                print(f"[INFO] Testing model for Scenarios {self.args.num_scenarios} | {self.args.sector} | {ticker_name} | Window {window_idx+1}")
                output_dir = f'./stock_prediction/pred_results/{model_name}/{self.args.GBM_type}/Train_{self.args.train_months}_Test_{self.args.sliding_test_months}/Scenarios_{self.args.num_scenarios}/{self.args.exp_root_path}/{self.args.sector}/{ticker_name}/'
                results, contribution_df = test_model(
                    model, test_loader, criterion, device, selected_scenarios, output_dir, window_idx+1, train_dataset, test_df_original
                )

                results_df = pd.DataFrame(results, columns=['Actual', 'Predicted'])
                os.makedirs(output_dir, exist_ok=True)
                results_df.to_csv(f'{output_dir}/window_{window_idx+1}.csv', index=False)