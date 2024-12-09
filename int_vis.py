import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go


@st.cache_data
def load_dataframe():
  df = pd.read_parquet('dados_agregados.parquet')
  return df


def update_layout(fig):
  # Atualize o layout com o switch button
  updt_fig = fig
  updt_fig = fig.update_layout(
      width=1200,
      height=600,
      title=dict(
          text='',
          font=dict(
              size=24  # Aumenta o tamanho do título
          ),
          x=0.5,
          y=0.95,
          xanchor='center'
      ),
      polar=dict(
          domain=dict(
              x=[0.1, 1],  # Ajusta a posição horizontal do radar
              y=[0.1, 1]   # Ajusta a posição vertical do radar
          ),
          angularaxis=dict(
              tickfont=dict(
                  size=20  # Aumenta o tamanho das labels das categorias
              ),
              tickangle=0  # Alinha as labels horizontalmente
          ),
          radialaxis=dict(
              tickfont=dict(
                  size=20  # Aumenta o tamanho das labels das escalas radiais
              ),
              range=[40, 70],
              tickmode='linear',
              dtick=10
          )
      ),
      legend=dict(
          font=dict(
              size=20  # Aumenta o tamanho das labels da legenda
          ),
          orientation='h',       # Define a orientação horizontal
          x=0.5,                 # Centraliza horizontalmente
          y=1.2,                # Posiciona abaixo do gráfico
          xanchor='center',      # Ancoragem horizontal
          yanchor='top'          # Ancoragem vertical
      ),
      showlegend=True,
      annotations=[
          dict(
              text=(
                  "As notas foram normalizadas e ajustadas para um intervalo de 0 a 100."
              ),
              showarrow=False,
              xref="paper",
              yref="paper",
              x=0.5,
              y=-0.1,  # Posiciona abaixo do gráfico
              xanchor='center',
              yanchor='top',
              font=dict(
                  size=16,
                  color="gray"
              ),
              align="center",
              width=800  # Ajusta a largura do texto
          ),
      ]
  )
  return updt_fig


def traces(fig, values, categories, name, line_color, fill_color, visible):
  new_fig = fig

  new_fig.add_trace(
      go.Scatterpolar(
          r=values,
          theta=categories,
          fill='toself',
          name=name,
          mode="lines",
          line=dict(
              color=line_color,
              width=4,
              ),
          visible=visible,
          opacity = 1,
          fillcolor = fill_color
      )
  )

  return new_fig

def calculate_mean(data, region):
  columns_of_interest = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
  new_data = data.loc[data.REGIAO == region].copy()
  average_by_gender = new_data.groupby('TP_SEXO')[columns_of_interest].mean()
  max_scores = data[columns_of_interest].max()

  return average_by_gender, max_scores


def grafico_lucas(df, filtro):
  regioes = {
      '1': 'norte',
      '2': 'nordeste',
      '3': 'sudeste',
      '4': 'sul',
      '5': 'centro-oeste'
  }

  df['REGIAO'] = df['CO_MUNICIPIO_PROVA'].astype(str).str[0]
  df['REGIAO'] = df['REGIAO'].map(regioes)

  results = {}

  for regiao in regioes.values():
    average_by_gender, max_scores = calculate_mean(df, regiao)
    normalized_means = average_by_gender / max_scores
    normalized_means_percent = normalized_means * 100
    results[regiao] = normalized_means_percent

  man = []
  woman = []
  for key, value in results.items():
    man.append((value.iloc[value.index == "M"].values.tolist()[0]))
    woman.append((value.iloc[value.index == "F"].values.tolist()[0]))

  categories = ['Natureza', 'Humanas', 'Linguagens', 'Matemática', 'Redação']
  regioes = ["Norte", "Nordeste", "Sul", "Sudeste", "Centro-Oeste"]
  homem = {}
  mulher = {}

  for i, regiao in enumerate(regioes):
    homem[regiao] = man[i]
    mulher[regiao] = woman[i]

  categories = categories + categories[:1]
  for regiao in regioes:
    homem[regiao] = homem[regiao] + homem[regiao][:1]
    mulher[regiao] = mulher[regiao] + mulher[regiao][:1]

  fig = go.Figure()

  fill_opacity = 0.2

  colors_rgb = {
      "blue": f"rgba(0, 0, 255, {fill_opacity})",
      "red": f"rgba(255, 0, 0, {fill_opacity})",
      "green": f"rgba(0, 255, 0, {fill_opacity})",
      "purple": f"rgba(128, 0, 128, {fill_opacity})",
      "orange": f"rgba(255, 165, 0, {fill_opacity})"
  }

  for regiao, key, value in zip(regioes, colors_rgb.keys(), colors_rgb.values()):

    if filtro == 'M':
      fig = traces(fig, homem[regiao], categories, regiao, line_color=key, fill_color=value, visible=True) 
    elif filtro == 'F':
      fig = traces(fig, mulher[regiao], categories, regiao, line_color=key, fill_color=value, visible=True)
    else:
      # traces(fig, homem[regiao], categories, regiao, line_color=key, fill_color=value, visible=True)
      # traces(fig, mulher[regiao], categories, regiao, line_color=key, fill_color=value, visible=True)
      geral = [(a + b) / 2 for a, b in zip(homem[regiao], mulher[regiao])]
      fig = traces(fig, geral, categories, regiao, line_color=key, fill_color=value, visible=True)
    
  fig = update_layout(fig)
  return fig


@st.cache_data
def grafico_beatriz(df):
  notas = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']

  # Calcular a média das notas por escolaridade do pai e da mãe
  media_pai_mae = df.groupby(['Q001', 'Q002'])[notas].mean().reset_index()

  pivot_table = media_pai_mae.pivot(index='Q001', columns='Q002', values='NU_NOTA_REDACAO')

  order = sorted(pivot_table.index)
  pivot_table = pivot_table.loc[order, order]

  # Criar o heatmap com a escala de cores personalizada
  fig = plt.figure(figsize=(8, 5))
  sns.heatmap(
      pivot_table,
      annot=True,               
      fmt=".1f",                
      cmap='RdYlGn',            # Colormap que vai de vermelho (low) a verde (high)
      cbar_kws={'label': 'Média Nota Redação'}  
  )
  plt.title(f'Relação entre Escolaridade dos Pais e Nota de Redação de alunos')
  plt.xlabel('Escolaridade da Mãe')
  plt.ylabel('Escolaridade do Pai')
  plt.show()
  return fig


def main():
  st.set_page_config(page_title='Dados - ENEM', layout="wide", page_icon='📚')
  st.title('Explorando os dados do ENEM')
  st.divider()

  st.subheader('Trabalho final - Visualização de Informações')
  st.markdown("- Universidade Federal de Goiás (UFG) - BCC \n - Integrantes: \n   - Beatriz Pinheiro de Lemos Lopes\n   - Lucas Braga Santos\n    - Victor Ferraz de Castro Ribeiro")
  st.divider()

  st.subheader('Perguntas exploradas nos gráficos')
  graf_option = st.selectbox('Qual gráfico deseja entender?', options=['Mapa de Calor', 'Gráfico de Radar', 'Ambos'],  index=2)  
  
  if graf_option == 'Ambos':
    st.markdown('''Através dos gráficos e com a edição dos filtros é possível entender como o conjunto varia de acordo com:
- Sexo (M ou F)
- Ano (2017, 2020 ou 2023)
- Tipo da escola (pública, privada ou estrangeira)
''')
  elif graf_option == 'Mapa de Calor':
    st.markdown('''Com o mapa de calor temos uma visualização completa da relação entre a nota da redação e a escolaridade dos pais,
  assim consegumos identificiar pontos como:
- Quais combinações de escolaridade dos pais resultam em melhores ou piores desempenhos na redação?
- Qual é a influência da escolaridade dos pais nas notas de redação dos participantes?''')
  elif graf_option == 'Gráfico de Radar':
    st.markdown('''Com o gráfico de radar temos uma representação esquemática do desempenho de toda as regiões do Brasil em relação às grandes áreas do enem,
  assim consegumos identificiar pontos como:''')

  st.divider()

  st.header('Filtros disponíveis')
  f1, f2, f3 = st.columns(3, vertical_alignment='center', gap='medium')

  with f1:
    filtro_ano = st.multiselect('Anos para análise', options=[2017, 2020, 2023], placeholder='Escolha o(s) ano(s)', default=[2017, 2020, 2023] )
  with f2:
    filtro_sexo = st.segmented_control('Sexo para análise', options=['M', 'F', 'Ambos'], default='Ambos')
  with f3:
    filtro_escola = st.segmented_control('Tipo de escola para análsie', options=['Pública', 'Privada', 'Estrangeira', 'Geral'], default='Geral')

  df = load_dataframe()

  df = df[df['NU_ANO'].isin(filtro_ano)]
  filter_df = df[df['NU_ANO'].isin(filtro_ano)].copy()
  
  if filtro_sexo != 'Ambos':
    filter_df = df[df['TP_SEXO'] == filtro_sexo]

  if filtro_escola != 'Geral':
    if filtro_escola == 'Privada':
      filter_df = filter_df[filter_df['TP_ESCOLA'] == 3]
      df = df[df['TP_ESCOLA'] == 3]
    elif filtro_escola == 'Pública':
      filter_df = filter_df[filter_df['TP_ESCOLA'] == 2]
      df = df[df['TP_ESCOLA'] == 2]
    else:
      filter_df = filter_df[filter_df['TP_ESCOLA'] == 4]
      df = df[df['TP_ESCOLA'] == 4]

  if filtro_ano == []:
      st.warning('Selecione algum ano!')
  else:

    beatriz, lucas = st.columns(2)

    with beatriz:
      st.header('Mapa de Calor')

      gb = grafico_beatriz(filter_df)
      st.pyplot(gb)

      st.subheader('Legenda para os eixos')
      col1, col2 = st.columns(2)

      with col1:
        st.markdown('**A**: Nunca estudou')
        st.markdown('**B**: Não completou a 4ª série/5º ano do Ensino Fundamental')
        st.markdown('**C**: Completou a 4ª série/5º ano, mas não completou a 8ª série/9º ano do Ensino Fundamental')
        st.markdown('**D**: Completou a 8ª série/9º ano do Ensino Fundamental, mas não completou o Ensino Médio')
      with col2:
        st.markdown('**E**: Completou o Ensino Médio, mas não completou a Faculdade')
        st.markdown('**F**: Completou a Faculdade, mas não completou a Pós-graduação')
        st.markdown('**G**: Completou a Pós-graduação')
        st.markdown('**H**: Não sei')

    with lucas:
      st.header('Gráfico de Radar')
      gl = grafico_lucas(df, filtro_sexo)
      st.plotly_chart(gl)
  return


if __name__ == '__main__':
  main()
