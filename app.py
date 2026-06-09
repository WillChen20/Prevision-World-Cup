import streamlit as st
import pandas as pd
import joblib
import kagglehub as kh
import os

# ========================================================
# CONFIGURAÇÃO DA PÁGINA DO APP (Visual Mobile-Friendly)
# ========================================================
st.set_page_config(
    page_title="Preditor Copa 2026", 
    page_icon="⚽", 
    layout="centered"
    #initial_sidebar_state="collapsed"
)

st.title("⚽ Simulador de Jogos - Copa 2026")
st.markdown("Preveja o placar de qualquer confronto utilizando Inteligência Artificial.")

# ========================================================
# 1. FUNÇÃO PARA CARREGAR O MODELO E OS DADOS
# ========================================================
@st.cache_resource
def carregar_sistema_ia():
    model_home = joblib.load('model_home.joblib')
    model_away = joblib.load('model_away.joblib')
    le_teams = joblib.load('le_teams.joblib')
    le_tourn = joblib.load('le_tourn.joblib')
    return model_home, model_away, le_teams, le_tourn

@st.cache_data
def carregar_dados_api_para_app():
    # Faz o download automático exato que eu fez no notebook
    path = kh.dataset_download("martj42/international-football-results-from-1872-to-2017")
    df_results = pd.read_csv(f"{path}/results.csv")
    df_shootouts = pd.read_csv(f"{path}/shootouts.csv")

    # Faz a união e filtros rápidos idênticos ao seu treino
    df_unido = pd.merge(df_results, df_shootouts[['date', 'home_team', 'away_team', 'winner']], 
                     on=['date', 'home_team', 'away_team'], how='left')

    df_unido['date'] = pd.to_datetime(df_unido['date'])
    
    # Filtra e mantém apenas os jogos que JÁ aconteceram para servir de histórico de forma
    df_historico = df_unido[df_unido['date'].dt.year >= 2014].dropna(subset=['home_score']).copy()
    return df_historico

# Executando as cargas automáticas
try:
    model_home, model_away, le_teams, le_tourn = carregar_sistema_ia()
    df_treino = carregar_dados_api_para_app() # Agora os dados vêm direto da API
    
    # Pegar a lista de times válidos em ordem alfabética para o Dropdown
    lista_times = sorted(list(le_teams.classes_))
    lista_torneios = sorted(list(le_tourn.classes_))
except Exception as e:
    st.error(f"Erro ao carregar os dados ou modelos: {e}")
    st.stop()
    
# ============================================================
# 2. FUNÇÃO AUXILIAR PARA CALCULAR A FORMA (Usando a base da API)
# ============================================================
# Nota: Para o app rodar leve em produção, você pode carregar o dataframe de treino pronto
def calcular_media_gols_10_app(team, df_completo):
    # Busca os últimos 10 jogos no dataframe que veio da API
    past_games = df_completo[((df_completo['home_team'] == team) | 
                              (df_completo['away_team'] == team))].sort_values('date', ascending=False).head(10)
    
    if past_games.empty:
        return 1.2 # Média caso seja um time sem histórico recente
    
    gols = [row['home_score'] if row['home_team'] == team else row['away_score'] for _, row in past_games.iterrows()]
    return sum(gols) / len(gols)

# ============================
# 3. INTERFACE VISUAL DO APP
# ============================
st.subheader("Configurar o Confronto")

# Dicionário de tradução/mapeamento rápido para os emojis funcionarem.
# Opcional: Opcionalmente, você pode usar emojis padrão ou mapear os principais.
# Para garantir que todo país tenha um ícone padrão se não mapeado, usamos uma função:
def obter_bandeira(nome_pais):
    # Mapeamento manual dos principais países da Copa para garantir precisão
    bandeiras = {
        "Argentina": "🇦🇷", "France": "🇫🇷", "Brazil": "🇧🇷", "Spain": "🇪🇸",
        "Portugal": "🇵🇹", "Germany": "🇩🇪", "England": "🇬🇧", "Italy": "🇮🇹",
        "Uruguay": "🇺🇾", "Belgium": "🇧🇪", "Netherlands": "🇳🇱", "Croatia": "🇭🇷",
        "Chile": "🇨🇱", "Colombes": "🇨🇴", "Peru": "🇵🇪", "Ecuador": "🇪🇨",
        "Japan": "🇯🇵", "South Korea": "🇰🇷", "Morocco": "🇲🇦", "Senegal": "🇸🇳",
        "United States": "🇺🇸", "Mexico": "🇲🇽", "Canada": "🇨🇦", "Jordan": "🇯🇴",
        "DR Congo": "🇨🇩", "South Africa": "🇿🇦", "Czechoslovakia": "🇨🇿", "Bosnia and Herzegovina": "🇧🇦",
        "Paraguay": "🇵🇾", "Qatar": "🇶🇦", "Sweden": "🇸🇪", "Switzerland": "🇨🇭",
        "Haiti": "🇭🇹", "Austria": "🇦🇹", "Scotland": "🏴󠁧󠁢󠁳󠁣󠁴󠁿", "Turkey": "🇹🇷", "Curaçao": "🏳️",
        "Ivory Coast": "🇨🇮", "Ghana": "🇬🇭", "Tunisia": "🇹🇳", "Cape Verde": "🇨🇻",
        "Algeria": "🇩🇿", "Egypt": "🇪🇬", "Saudi Arabia": "🇸🇦", "Iran": "🇮🇷", "Iraq": "🇮🇶",
        "Iraq": "🇮🇶", "Norway": "🇳🇴", "New Zealand": "🇳🇿", "Panama": "🇵🇦", "Uzbekistan": "🇺🇿"
    }
    # Retorna a bandeira mapeada ou uma bola de futebol genérica caso o país seja muito alternativo
    return bandeiras.get(nome_pais, "⚽")

# Criamos uma lista de opções visual que junta a Bandeira + Nome do País
opcoes_com_bandeira = [f"{obter_bandeira(pais)} {pais}" for pais in lista_times]

col1, col2 = st.columns(2)
with col1:
    # O usuário visualiza a bandeira, mas o index nos ajuda a achar o país original
    selecao_casa = st.selectbox("Seleção da Casa(Mandante)", 
                                opcoes_com_bandeira, 
                                index=0)
    # Removemos a bandeira para obter o nome puro do país
    time_casa = selecao_casa.split(" ", 1)[1]
    
with col2:
    selecao_fora = st.selectbox("Seleção de Fora(Visitante)", 
                                opcoes_com_bandeira, 
                                index=1)
    time_fora = selecao_fora.split(" ", 1)[1]
    
# Opções adicionais recolhidas para o design ficar limpo no celular
with st.expander("Configurações Avançadas (Clique para expandir)"):
    torneio_selecionado = st.selectbox("Tipo de Torneio", lista_torneios, index=[i for i, t in enumerate(lista_torneios) if t == "FIFA World Cup"][0] if "FIFA World Cup" in lista_torneios else 0)
    campo_neutro = st.checkbox("Jogo em Campo Neutro?", value=True)
    
# ==============================
# 4. BOTÃO PARA PREVER O PLACAR
# ==============================
if st.button("🚀 Simular Partida", use_container_width=True):
    if time_casa == time_fora:
        st.warning("Por favor, selecione times diferentes para casa e fora.")
    else:
        # Processar IDs usando o nome limpo sem a bandeira
        id_casa = le_teams.transform([time_casa])[0]
        id_fora = le_teams.transform([time_fora])[0]
        id_torneio = le_tourn.transform([torneio_selecionado])[0]
        
        # Calcular a forma usando a função que agora usa o dataframe da API
        forma_casa = calcular_media_gols_10_app(time_casa, df_treino)
        forma_fora = calcular_media_gols_10_app(time_fora, df_treino)
        
        # Estruturar dados para a IA
        dados_jogo = pd.DataFrame([[id_casa, id_fora, id_torneio, int(campo_neutro), forma_casa, forma_fora]],
                                  columns=['home_id', 'away_id', 'tournament_id', 'neutral', 'home_form', 'away_form'])
        
        # Predição
        gol_casa = round(model_home.predict(dados_jogo)[0], 2)
        gol_fora = round(model_away.predict(dados_jogo)[0], 2)
        
        # Exibir resultado
        st.success("### Resultado da Simulação:")
        
        c1, c2, c3 = st.columns([3, 1, 3])
        with c1:
            st.metric(label=f"{obter_bandeira(time_casa)} {time_casa}", value=f"{gol_casa} gols")
        with c2:
            st.markdown("<h2 style='text-align: center;'>vs</h2>", unsafe_allow_html=True)
        with c3:
            st.metric(label=f"{obter_bandeira(time_fora)} {time_fora}", value=f"{gol_fora} gols")
            
        # Lógica de Tendência / Palpite do Gestor
        st.markdown("---")
        if gol_casa > gol_fora + 0.4:
            st.info(f"💡 **Análise da IA:** Forte tendência de vitória para o **{time_casa}**.")
        elif gol_fora > gol_casa + 0.4:
            st.info(f"💡 **Análise da IA:** Forte tendência de vitória para o **{time_fora}**.")
        else:
            st.info("💡 **Análise da IA:** Equilíbrio tático extremo. Jogo com alta probabilidade de Empate.")