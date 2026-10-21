Burgers PINN experiments
========================

Run each model in a separate Python process:

  python run_pinn_tanh.py
  python run_pinn_pau.py
  python run_pinn_smpaf.py

Outputs:

  results/pinn_tanh.npz
  results/pinn_pau.npz
  results/pinn_smpaf.npz

All scripts use the same:
- Burgers problem with alpha=1.0;
- architecture and initial network weights;
- fixed train/validation collocation points;
- number of weight updates and learning rate;
- reference numerical solution.

The NPZ files store loss histories, train/validation points, final network
parameters, reference/predicted solutions, relative L2 error and runtime.
No plotting is done during training.

Memory note:
PAU/smPAF use a first-order bilevel coefficient update (theta is frozen during
the validation coefficient update) instead of differentiating through the
entire unrolled inner optimization. This is intentional to lower memory use for
the second-order PINN residual.


Execucao automatizada E1-E4 em um unico arquivo
----------------------------------------------
Arquivo: run_experiments.py (usa apenas o pinn_helpers.py desta pasta).

Dentro de pinn_burgers_experiments, escolha o experimento:
  python run_experiments.py E1
  python run_experiments.py E2
  python run_experiments.py E3
  python run_experiments.py E4

A partir da raiz do projeto:
  python pinn_burgers_experiments/run_experiments.py E1

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

Resultados sempre em pinn_burgers_experiments/results/full/ (ou results/smoke/):
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
