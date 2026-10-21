Experimentos PINN para Korteweg-de Vries (KdV)
=============================================

Equacao: u_t + 6 u u_x + beta u_xxx = 0, beta = 1.
Dominio: x em [-10, 10], t em [0, 1].
Solucao exata: u(x,t) = (c/2) sech^2(sqrt(c/beta)*(x-c*t-x0)/2),
com c = 1 e x0 = -2. A condicao inicial e essa solucao em t=0.
Os contornos usam os valores exatos de u nas duas extremidades e de
u_x na extremidade direita (tres restricoes para a derivada espacial terceira).
Os valores nos contornos nao sao aproximados por zero nem periodicos.

A partir da raiz do projeto:
  python pinn_KV_experiments/run_pinn_tanh.py
  python pinn_KV_experiments/run_pinn_pau.py
  python pinn_KV_experiments/run_pinn_smpaf.py

Tambem e possivel executar dentro desta pasta. As saidas sempre ficam em:
  results/pinn_kdv_tanh_exp1.npz
  results/pinn_kdv_pau_exp1.npz
  results/pinn_kdv_smpaf_exp1.npz
Altere EXPERIMENT nos scripts para preservar execucoes anteriores.

Dependencias: Python, NumPy e JAX (com o backend adequado ao seu equipamento).
Configuracoes do problema, arquitetura, amostragem e treino: pinn_helpers.py.
As entradas sao normalizadas para [-1,1]; as derivadas em coordenadas fisicas
incluem os fatores da regra da cadeia via autodiferenciacao.
Os tres modelos compartilham pontos fixos, pesos iniciais, arquitetura e
learning rate. PAU usa sempre um passo interno por passo externo (8000
atualizacoes por padrao), inclusive no runner de experimentos.
Os demais modelos usam OUTER_STEPS * INNER_STEPS atualizacoes.
Por padrao sao 8000 * 40 nos demais modelos; derivadas de terceira ordem podem
exigir bastante tempo e memoria. Reduza esses valores para testes iniciais.
PAU/smPAF atualizam coeficientes pela perda de validacao, com pesos congelados,
sem diferenciar atraves de todo o ciclo interno (bilevel de primeira ordem).

Os NPZ incluem historicos de perdas, pontos de treino/validacao (incluindo
contorno de derivada), parametros, coeficientes adaptativos, configuracao JSON,
grade, referencia exata, previsao, erro L2 relativo e tempo de treino.
A referencia e analitica; nao se utiliza o solver de Burgers.
Nao ha geracao de graficos durante o treinamento.

A PAU original usa abs no denominador e nao e suave nos zeros de Q.
Ela foi mantida para comparacao, mas autodiferenciacao nesses pontos nao
representa uma terceira derivada classica; tanh e smPAF sao suaves.
Convergencia deve ser avaliada pelos historicos e pelo erro L2 de cada treino.

Referencia da equacao e dos solitons:
https://lie.ac.brocku.ca/~sanco/solitons/kdv_solitons.php


Execucao automatizada E1-E4 em um unico arquivo
----------------------------------------------
Arquivo: run_experiments.py (usa apenas o pinn_helpers.py desta pasta).

Dentro de pinn_KV_experiments, escolha o experimento:
  python run_experiments.py E1
  python run_experiments.py E2
  python run_experiments.py E3
  python run_experiments.py E4

A partir da raiz do projeto:
  python pinn_KV_experiments/run_experiments.py E1

Teste curto e validacao:
  python run_experiments.py E1 --smoke --seed 5
  python run_experiments.py --validate

Configuracao individual e previa sem treinar:
  python run_experiments.py E2 --inner-steps 5 --seed 5
  python run_experiments.py E3 --kappa 0.01 --seed 5
  python run_experiments.py E4 --lr-coeffs 0.001 --seed 5
  python run_experiments.py E1 --dry-run

Somente para executar todos explicitamente:
  python run_experiments.py all

Resultados sempre em pinn_KV_experiments/results/full/ (ou results/smoke/):
  E1/ ... E4/     referencias aos treinos, com reuso dos baselines
  runs/<id>/     config.json, metrics.json, loss_history.npz, predictions.npz,
                 parameters.npz, initial_parameters.npz, datasets.npz,
                 checkpoint.npz e complete.json
  failures/      falhas e parametros no momento da divergencia
  summary/       tabelas CSV e graficos PDF/PNG

E1 compara tanh, ReLU, PAU e smPAF com seeds 5 a 9. E2 varia passos internos
(1,5,20,40), E3 varia kappa (1e-4,1e-2,0.3,1), e E4 varia LR dos coeficientes
(1e-5,1e-4,5e-4,1e-3). E2-E4 usam smPAF. Baselines identicos sao reutilizados.
A configuracao completa preserva a arquitetura (2,64,164,64,1), 8000 passos
externos e 40 internos no baseline. Kappa padrao: 0.3. Os testes --smoke
usam explicitamente rede/dados menores e dois passos externos.
E2 mede conjuntamente frequencia e orcamento: 8000 passos externos resultam
em 8000,40000,160000,320000 atualizacoes de pesos; nao ha normalizacao.

Repetir o comando retoma checkpoints ou reutiliza resultados completos.
Use --retry-failed para repetir falhas, --summarize-only para gerar tabelas e
graficos, --results CAMINHO para outra saida e --no-plots para omitir figuras.
As estatisticas excluem falhas/incompletos e usam desvio padrao amostral.
O melhor valor de validacao e o minimo dos pontos registrados, incluindo
o passo inicial. A solucao de referencia e usada somente apos o treino.

Pesos iniciais sao pareados por seed; pontos de treino/validacao usam sempre
as seeds 5/6. Coeficientes usam chave independente e inicializacao original.
Tanh/ReLU: 21477 parametros. PAU/smPAF: 21477 pesos/vieses + 2628 coeficientes.
Derivadas, perdas, SGD e bilevel de primeira ordem sao os do pinn_helpers.py.
A temporizacao sincroniza JAX e exclui compilacao, checkpoint I/O e avaliacao.
O runner usa locks POSIX para evitar gravacoes simultaneas do mesmo treino.

Limitacoes preservadas: ReLU tem derivadas espaciais de ordem >=2 nulas
quase em todo lugar; PAU nao e suave nos zeros de Q. O inicializador original
com 9 coeficientes favorece o termo cubico, apesar de um comentario antigo
mencionar identidade. Nada disso e corrigido alterando a matematica.
Os scripts antigos mantem seus defaults; este runner aplica kappa=0.3 e
40 passos internos no baseline, conforme solicitado.
