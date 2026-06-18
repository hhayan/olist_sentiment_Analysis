Mounted at /content/drive
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 33491 entries, 0 to 33490
Data columns (total 3 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   review_score            33491 non-null  int64 
 1   review_comment_message  33491 non-null  object
 2   product_category_name   33491 non-null  object
dtypes: int64(1), object(2)
memory usage: 785.1+ KB
72
review_score
1     6800
2     1864
3     3099
4     4916
5    16812
Name: count, dtype: int64
비율:review_score
1    20.30
2     5.57
3     9.25
4    14.68
5    50.20
Name: count, dtype: float64
불균형 비율: 9.02배
클래스 불균형 → class_weight 필요
최대 길이: 208
레이블 매핑 확인
review_score 1 → sentiment 0
review_score 2 → sentiment 1
review_score 3 → sentiment 2
review_score 4 → sentiment 3
review_score 5 → sentiment 4
Train: 23,443개 (70.0%)
Val:   5,024개 (15.0%)
Test:  5,024개 (15.0%)
Train 클래스 분포:
sentiment
0     4760
1     1305
2     2169
3     3441
4    11768
Name: count, dtype: int64
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
tokenizer_config.json: 
 1.20k/? [00:00<00:00, 97.6kB/s]
vocab.txt: 
 996k/? [00:00<00:00, 32.8MB/s]
tokenizer.json: 
 2.92M/? [00:00<00:00, 71.7MB/s]
special_tokens_map.json: 100%
 125/125 [00:00<00:00, 13.6kB/s]
토큰 길이 통계:
평균: 22.7
95%: 54
99%: 63
max_length: 128

이모지 포함 여부
총 리뷰:        5,024개
이모지 포함:    35개 (0.70%)
이모지 없음:    4,989개 (99.30%)
============================================================
이모지 포함 리뷰 샘플 (35개 중 10개)
============================================================

1. 감성: 4 | 예측: N/A
   Ótimo e foi entregue antes da data😀...

2. 감성: 0 | 예측: N/A
   Espero ainda receber o produto🙏🏻...

3. 감성: 0 | 예측: N/A
   Comprei uma cortina de presente para minha mãe, mas veio outro tipo de cortina inclusive bem inferior e alguns produtos que eu não precisava. Só não d...

4. 감성: 3 | 예측: N/A
   Adorei o produto, estou muito satisfeita, foi entregue antes do prazo!!😉...

5. 감성: 4 | 예측: N/A
   Super indico tanto a loja como o prpduto máquina leve sem barulho trabalho perfeito entregue antes do prazo amei 👏👏👏👏👏...

6. 감성: 2 | 예측: N/A
   É bem bonita e bem acabada mais pena q veio a almofadinha d uma cor e a casinha de outra 😢
Por esse motivo nao vou enche de estrelas 😭...

7. 감성: 4 | 예측: N/A
   Produto de ótima qualidade, adorei sem falar que fiz o pedido em um dia e já chegou no outro !!!! Super recomendo😊...

8. 감성: 2 | 예측: N/A
   Ótima compra chegou bem antes do prazo, ainda não usei mas fiquei hiper feliz recomendo 👏👏👏👏👏...

9. 감성: 4 | 예측: N/A
   Entrega antes do prazo previsto 👍...

10. 감성: 3 | 예측: N/A
   Amei muito lindas e ótimo material , chegou muito antes do prazo 😍...

Test set: 5,024개
클래스 분포:
sentiment
0    1020
1     280
2     465
3     737
4    2522
Name: count, dtype: int64
config.json: 100%
 851/851 [00:00<00:00, 38.6kB/s]
model.safetensors: 100%
 541M/541M [00:06<00:00, 216MB/s]
추론 중: 100%|██████████| 157/157 [00:10<00:00, 15.49it/s]

추론 완료: 5,024개
Baseline 성능
Accuracy:        0.1943
F1 (Weighted):   0.1891
F1 (Macro):      0.1936

목표 F1 (Weighted): 0.2391 (+5%)
클래스별 성능
               precision    recall  f1-score   support

Very Negative     0.5153    0.1647    0.2496      1020
     Negative     0.1182    0.5714    0.1958       280
      Neutral     0.0892    0.2000    0.1233       465
     Positive     0.1614    0.4478    0.2373       737
Very Positive     0.8755    0.0892    0.1619      2522

     accuracy                         0.1943      5024
    macro avg     0.3519    0.2946    0.1936      5024
 weighted avg     0.5826    0.1943    0.1891      5024

Baseline F1 (Weighted): 0.1891
목표 F1 (LoRA 후):      0.2391
개선 필요량:            +0.0500 (26.4%)
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 5024 entries, 0 to 5023
Data columns (total 4 columns):
 #   Column                  Non-Null Count  Dtype 
---  ------                  --------------  ----- 
 0   review_score            5024 non-null   int64 
 1   review_comment_message  5024 non-null   object
 2   product_category_name   5024 non-null   object
 3   sentiment               5024 non-null   int64 
dtypes: int64(2), object(2)
memory usage: 157.1+ KB

Train: 23,443개
Val:   5,024개

Train 클래스 분포:
sentiment
0     4760
1     1305
2     2169
3     3441
4    11768
Name: count, dtype: int64

Class Weights:
  클래스 0: 0.98 (샘플 4760개)
  클래스 1: 3.59 (샘플 1305개)
  클래스 2: 2.16 (샘플 2169개)
  클래스 3: 1.36 (샘플 3441개)
  클래스 4: 0.40 (샘플 11768개)
Map: 100%
 23443/23443 [00:07<00:00, 3992.23 examples/s]
Map: 100%
 5024/5024 [00:01<00:00, 4321.80 examples/s]

✅ 토큰화 완료
trainable params: 741,893 || all params: 136,070,410 || trainable%: 0.5452


Step	Training Loss	Validation Loss	Macro F1	Weighted F1	Recall 0	Recall 1	Recall 2	Recall 3	Recall 4
200	1.327100	1.296720	0.386000	0.582991	0.164706	0.571429	0.200000	0.447761	0.089215
400	1.259300	1.295293	0.382986	0.600401	0.164706	0.571429	0.200000	0.447761	0.089215
600	1.262700	1.240564	0.419476	0.563066	0.164706	0.571429	0.200000	0.447761	0.089215
800	1.207500	1.222720	0.416058	0.563784	0.164706	0.571429	0.200000	0.447761	0.089215
1000	1.202500	1.227779	0.438098	0.623415	0.164706	0.571429	0.200000	0.447761	0.089215
1200	1.200700	1.243046	0.416465	0.617544	0.164706	0.571429	0.200000	0.447761	0.089215
1400	1.163800	1.216607	0.430409	0.617249	0.164706	0.571429	0.200000	0.447761	0.089215
1600	1.141100	1.226953	0.438254	0.620937	0.164706	0.571429	0.200000	0.447761	0.089215

Test Set 최종 평가
Map: 100%
 5024/5024 [00:00<00:00, 9056.18 examples/s]

Accuracy:        0.6164
Weighted F1:     0.6149
Macro F1:        0.4294

클래스별 Recall:
  클래스 0: 0.6314
  클래스 1: 0.2821
  클래스 2: 0.2731
  클래스 3: 0.1682
  클래스 4: 0.8418
Baseline vs LoRA 비교

⭐ Recall(4) 개선:
  Baseline: 0.0892
  LoRA:     0.8418
  개선량:   +0.7526 (843.7%)

✅ 목표 달성! (Recall 4 ≥ 0.40)

클래스별 Recall:
  클래스 0: 0.6314
  클래스 1: 0.2821
  클래스 2: 0.2731
  클래스 3: 0.1682
  클래스 4: 0.8418

======================================================================
📊 성능 지표 해석
======================================================================

✅ Weighted F1 높음 (0.615)
   → 실사용 성능 우수
   → 다수 클래스(Very Positive) 잘 예측하여 전체 커버리지 ↑
   → 고객 만족도 모니터링 정확도: 74.7%

⚠️  Macro F1 낮음 (0.429)
   → 소수 클래스(Negative, Neutral, Positive) 여전히 어려움
   → 클래스 간 성능 불균형 존재
   → 특히 Neutral(클래스 2) 가장 취약: 27.3%

📈 개선 요약
   Baseline Weighted F1: 0.1891
   LoRA Weighted F1:     0.6149
   개선량:               +0.4258 (225.2%)

✅ 분석 데이터: 5,024개
카테고리 수: 69개
1️⃣ 전체 부정 키워드 Top 10
 1. produto               698회
 2. recebi                349회
 3. comprei               229회
 4. veio                  225회
 5. entrega               161회
 6. ainda                 158회
 7. chegou                153회
 8. entregue              141회
 9. estou                 120회
10. muito                 119회

======================================================================
2️⃣ 부정 비율 높은 카테고리 Top 5
1. construcao_ferramentas_seguranca         41.7% (5/12)
2. climatizacao                             38.9% (7/18)
3. eletrodomesticos                         36.7% (11/30)
4. moveis_escritorio                        34.8% (32/92)
5. audio                                    33.3% (5/15)

======================================================================
3️⃣ 각 카테고리별 부정 키워드 Top 3

1. construcao_ferramentas_seguranca
   1) recebi            2회
   2) central           2회
   3) veja              2회

2. climatizacao
   1) produto           3회
   2) recebi            2회
   3) ainda             2회

3. eletrodomesticos
   1) produto           6회
   2) original          5회
   3) comprei           3회

4. moveis_escritorio
   1) produto          12회
   2) cadeira          11회
   3) veio             10회

5. audio
   1) produto           6회
   2) recebi            3회
   3) hello             2회
/tmp/ipython-input-2640576696.py:44: DeprecationWarning: DataFrameGroupBy.apply operated on the grouping columns. This behavior is deprecated, and in a future version of pandas the grouping columns will be excluded from the operation. Either pass `include_groups=False` to exclude the groupings or explicitly select the grouping columns after groupby to silence this warning.
  cat_neg = test_result.groupby('product_category_name').apply(

✅ 예측 확률 저장 완료
평균 확률 차이: 0.778

======================================================================
📊 Confidence 분포 분석
======================================================================

정답 평균 확률:   2.311
오답 평균 확률:   1.292
차이:            1.019

Low Confidence (<0.5): 376개 (7.5%)
애매한 샘플 (<0.1 차이): 515개 (10.3%)

======================================================================
🔍 애매한 샘플 직접 확인 (확률 차이 < 0.1)
======================================================================

1. ❌ | 실제: Neg | 예측: Very Neg
   확률: 1.68 (2등: 1.62, 차이: 0.06)
   리뷰: O prazo de entrega não foi cumprida e só comprei para dar de presente para uma mãe que está prestes a ganhar bebê. Só co...
   카테고리: cool_stuff

2. ✅ | 실제: Very Neg | 예측: Very Neg
   확률: 1.47 (2등: 1.42, 차이: 0.04)
   리뷰: O produto veio com defeito enviei email e nao obtive resposta...
   카테고리: telefonia

3. ✅ | 실제: Neg | 예측: Neg
   확률: 0.64 (2등: 0.61, 차이: 0.03)
   리뷰: Entrei em contato com o Vendedor ((RPGrupo) que prometeu a substituição. Como o prazo para avaliação pelas lannister ter...
   카테고리: informatica_acessorios

4. ❌ | 실제: Very Neg | 예측: Neutral
   확률: 0.20 (2등: 0.13, 차이: 0.06)
   리뷰: nao...
   카테고리: moveis_decoracao

5. ✅ | 실제: Very Pos | 예측: Very Pos
   확률: 0.48 (2등: 0.45, 차이: 0.03)
   리뷰: ATENDEU AO PROPÓSITO...
   카테고리: cama_mesa_banho

6. ❌ | 실제: Neutral | 예측: Very Neg
   확률: 0.17 (2등: 0.07, 차이: 0.10)
   리뷰: Sem opinião....
   카테고리: esporte_lazer

7. ❌ | 실제: Pos | 예측: Very Neg
   확률: -0.00 (2등: -0.02, 차이: 0.02)
   리뷰: Sem restrição pra compra....
   카테고리: brinquedos

8. ❌ | 실제: Very Neg | 예측: Neg
   확률: -0.21 (2등: -0.30, 차이: 0.08)
   리뷰: PRODUTO PARECE QUE JA FOI UTILIZADO FEIO TUDO SUJO E RISCADO...
   카테고리: portateis_cozinha_e_preparadores_de_alimentos

9. ❌ | 실제: Very Pos | 예측: Pos
   확률: 0.53 (2등: 0.44, 차이: 0.09)
   리뷰: Entrega no prazo, produto batia com a descrição. Adorei...
   카테고리: moveis_decoracao

10. ❌ | 실제: Very Neg | 예측: Pos
   확률: 0.16 (2등: 0.07, 차이: 0.10)
   리뷰: Sem comentários....
   카테고리: informatica_acessorios

======================================================================
⚠️ 자주 틀리는 클래스 조합 (Top 10)
======================================================================

실제           예측           빈도       평균 확률     
---------------------------------------------
Pos          Very Pos     375      2.358
Very Neg     Neg          244      1.103
Very Pos     Pos          209      1.002
Pos          Neutral      149      0.822
Neutral      Neg          119      1.028
Neutral      Very Neg     106      1.452
Neg          Very Neg     103      1.444
Very Pos     Neutral      89       0.722
Very Neg     Neutral      86       0.569
Neg          Neutral      74       0.718

======================================================================
🏷️  액션 태그 분류
======================================================================

액션 태그별 분포:
  정답                    3000개 (59.7%)
  재라벨링_후보               1338개 (26.6%)
  전처리_개선                 376개 (7.5%)
  일반_오류                  278개 (5.5%)
  규칙_보완_후보                32개 (0.6%)

======================================================================
📋 액션별 샘플 예시
======================================================================

🔹 재라벨링_후보 (1338개)

  1. 실제: Pos | 예측: Very Pos | 확률: 1.28
     Costumo receber os itens bem antes do prazo dado pelo site. mas como o Correio anda atrasado, desta ...

  2. 실제: Neg | 예측: Neutral | 확률: 1.21
     o mouse que eu queria veio errado, pois as especificações não são claras, eu pensei que o mouse era ...

  3. 실제: Very Pos | 예측: Neutral | 확률: 0.97
     O produto já chegou em casa um pouco demorado...

🔹 전처리_개선 (376개)

  1. 실제: Very Pos | 예측: Neutral | 확률: 0.30
     ?...

  2. 실제: Very Pos | 예측: Very Pos | 확률: 0.25
     Proteção máxima e livre de oleo...

  3. 실제: Very Pos | 예측: Very Neg | 확률: 0.38
     SEMPRE COMPRO E RECOMENDO ESTA LOJA, POIS SEMPRE RECEBI MINHAS COMPRAS ATÉ MESMO ANTES DO PRAZO....

🔹 규칙_보완_후보 (32개)

  1. 실제: Very Pos | 예측: Neg | 확률: 0.70
     O atraso foi culpa dos correios....

  2. 실제: Very Neg | 예측: Pos | 확률: 0.79
     Só ficou a deseja pelo fato de pagar pelo frete é ter que ir pega nos Correios....

  3. 실제: Very Pos | 예측: Very Neg | 확률: 0.59
     Ainda não utilizei as embalagens to aguardando a máquina chegar...

======================================================================
💡 오류 분석 인사이트
======================================================================

1️⃣ 재라벨링 필요 (고확신 오답)
   샘플 수: 1338개
   → 데이터 품질 검토 필요

2️⃣ 전처리 개선 (저확신)
   샘플 수: 376개
   → 짧은 리뷰, 애매한 표현 처리 개선

3️⃣ 규칙 보완 (극단 오분류)
   샘플 수: 32개
   → 비꼬는 표현, 혼합 감정 처리 필요

4️⃣ 하이퍼파라미터 조정 방향
   → 데이터 증강 우선 (특히 클래스 1, 2, 3)
   → Class Weight 재조정

5️⃣ 가장 많은 오류
   Pos → Very Pos: 375건
   → 해당 조합 집중 분석 필요

======================================================================
✅ Phase 5A 완료
======================================================================

✅ 분석 결과 저장: /content/drive/MyDrive/브라질 이커머스/NLP/processed/results/error_analysis.json


======================================================================
🔍 재라벨링 후보 - 데이터 품질 검토 필요 (총 1338개 중 10개)
======================================================================

[1] ❌ | 확률: 1.28 | 차이: 0.11
    실제: Pos          → 예측: Very Pos
    📝 Costumo receber os itens bem antes do prazo dado pelo site. mas como o Correio anda atrasado, desta vez foi no último dia previsto.
    📦 cama_mesa_banho

[2] ❌ | 확률: 1.21 | 차이: 0.79
    실제: Neg          → 예측: Neutral
    📝 o mouse que eu queria veio errado, pois as especificações não são claras, eu pensei que o mouse era sem fio, porém o mesmo é com fio. Mas dentro do qu
    📦 informatica_acessorios

[3] ❌ | 확률: 0.97 | 차이: 0.43
    실제: Very Pos     → 예측: Neutral
    📝 O produto já chegou em casa um pouco demorado
    📦 eletronicos

[4] ❌ | 확률: 1.58 | 차이: 0.01
    실제: Neutral      → 예측: Neg
    📝 Não foi entregue o mobile de carrinho.
    📦 bebes

[5] ❌ | 확률: 2.49 | 차이: 0.61
    실제: Pos          → 예측: Very Pos
    📝 Produto atendeu as expectativas e cumpre o prometido.
    📦 informatica_acessorios

[6] ❌ | 확률: 1.04 | 차이: 0.10
    실제: Very Neg     → 예측: Neg
    📝 Eu qero meu produto logo, por q essa demora !
    📦 beleza_saude

[7] ❌ | 확률: 0.85 | 차이: 0.09
    실제: Very Neg     → 예측: Neutral
    📝 Demora muito
    📦 cool_stuff

[8] ❌ | 확률: 1.60 | 차이: 0.40
    실제: Very Pos     → 예측: Pos
    📝 Tudo bom
    📦 cama_mesa_banho

[9] ❌ | 확률: 0.85 | 차이: 0.47
    실제: Neutral      → 예측: Neg
    📝 uma das bonecas veio sem chupeta
    📦 brinquedos

[10] ❌ | 확률: 1.04 | 차이: 0.45
    실제: Neutral      → 예측: Neg
    📝 O produto veio de acordo com o descrito no anúncio, exceto pelas 4 presilhas que prendem no banco, elas são muito fracas!! Uma pinscher conseguiu como
    📦 pet_shop

======================================================================
🔍 전처리 개선 - 짧은 리뷰/애매한 표현 (총 376개 중 10개)
======================================================================

[1] ❌ | 확률: 0.30 | 차이: 0.16
    실제: Very Pos     → 예측: Neutral
    📝 ?
    📦 cama_mesa_banho

[2] ✅ | 확률: 0.25 | 차이: 0.13
    실제: Very Pos     → 예측: Very Pos
    📝 Proteção máxima e livre de oleo
    📦 beleza_saude

[3] ❌ | 확률: 0.38 | 차이: 0.33
    실제: Very Pos     → 예측: Very Neg
    📝 SEMPRE COMPRO E RECOMENDO ESTA LOJA, POIS SEMPRE RECEBI MINHAS COMPRAS ATÉ MESMO ANTES DO PRAZO.
    📦 perfumaria

[4] ❌ | 확률: -0.03 | 차이: 0.04
    실제: Very Pos     → 예측: Pos
    📝 Adorei meu porta chaves
    📦 moveis_decoracao

[5] ❌ | 확률: 0.23 | 차이: 0.06
    실제: Very Pos     → 예측: Pos
    📝 Mesmo sabendo da data de entregue dita no pedido, tinha certeza que chegaria antes. Vcs são pontuais.
    📦 utilidades_domesticas

[6] ✅ | 확률: 0.31 | 차이: 0.05
    실제: Neg          → 예측: Neg
    📝 Vcs me mandaram numeração errada
    📦 fashion_calcados

[7] ❌ | 확률: 0.49 | 차이: 0.19
    실제: Very Pos     → 예측: Pos
    📝 não posso avaliar o produto pois ainda nao usei, mas os prazos foram cumpridos na entrega
    📦 cool_stuff

[8] ❌ | 확률: 0.33 | 차이: 0.25
    실제: Neutral      → 예측: Neg
    📝 Por favor preciso do estorno do valor total da compra.
    📦 telefonia

[9] ❌ | 확률: 0.35 | 차이: 0.05
    실제: Very Neg     → 예측: Neg
    📝 A descrição do produto fala q serve para veículos a parti de 2007 O que não é vdd tenho q fazer a troca do produto só serve para modelo 2009 pra frent
    📦 automotivo

[10] ✅ | 확률: 0.27 | 차이: 0.13
    실제: Very Neg     → 예측: Very Neg
    📝 DEMORA NA ENTREGA ,PRODUTO RUIM RESUMO NÃO GOSTEI
    📦 beleza_saude

======================================================================
🔍 규칙 보완 - 비꼬는 표현/혼합 감정 (총 32개 중 10개)
======================================================================

[1] ❌ | 확률: 0.70 | 차이: 0.36
    실제: Very Pos     → 예측: Neg
    📝 O atraso foi culpa dos correios.
    📦 cama_mesa_banho

[2] ❌ | 확률: 0.79 | 차이: 0.02
    실제: Very Neg     → 예측: Pos
    📝 Só ficou a deseja pelo fato de pagar pelo frete é ter que ir pega nos Correios.
    📦 climatizacao

[3] ❌ | 확률: 0.59 | 차이: 0.21
    실제: Very Pos     → 예측: Very Neg
    📝 Ainda não utilizei as embalagens to aguardando a máquina chegar
    📦 ferramentas_jardim

[4] ❌ | 확률: 0.56 | 차이: 0.00
    실제: Neg          → 예측: Very Pos
    📝 FALTOU CADEIRAS
    📦 moveis_escritorio

[5] ❌ | 확률: 0.66 | 차이: 0.45
    실제: Very Pos     → 예측: Very Neg
    📝 Ainda não testei o produto
    📦 beleza_saude

[6] ❌ | 확률: 0.71 | 차이: 0.16
    실제: Very Pos     → 예측: Neg
    📝 Amei o produto! Veio do jeito que esperávamos, porém o correio deixou a desejar! Duas vezes não esperou descer para fazer a entrega, obrigando a ir bu
    📦 cool_stuff

[7] ❌ | 확률: 0.77 | 차이: 0.10
    실제: Very Pos     → 예측: Neg
    📝 eu tinha aberto a reclamação solicitando o reenvio ontem e durante a tarde aconteceu

grato
    📦 utilidades_domesticas

[8] ❌ | 확률: 0.54 | 차이: 0.54
    실제: Very Neg     → 예측: Very Pos
    📝 Interponde entrega foi bem grande e mesmo assim nanotecnologia o produto.
Estou acompanhando e ainda nem chegou no Rio.
    📦 informatica_acessorios

[9] ❌ | 확률: 0.73 | 차이: 0.04
    실제: Very Pos     → 예측: Neg
    📝 Boa tarde...

Recebi sim meu produto, mas a lancheira deu problema na alça ela descosturou, como faço para trocar...
    📦 papelaria

[10] ❌ | 확률: 0.65 | 차이: 0.22
    실제: Very Pos     → 예측: Neg
    📝 Recebi só o kit ficou faltando um molinete de 3 rolamento que comprei juntos no mesmo dia por favor me manda resposta quando o outro chega
    📦 esporte_lazer

======================================================================
🔍 Very Pos → Pos 오류 (566건) (총 209개 중 10개)
======================================================================

[1] ❌ | 확률: 1.45 | 차이: 0.01
    실제: Very Pos     → 예측: Pos
    📝 Muito bom comprar nas lannister.
    📦 eletronicos

[2] ❌ | 확률: 0.79 | 차이: 0.41
    실제: Very Pos     → 예측: Pos
    📝 Eu só compro neste site tenho segurança ok
    📦 esporte_lazer

[3] ❌ | 확률: 0.59 | 차이: 0.17
    실제: Very Pos     → 예측: Pos
    📝 bom serviço
ainda estou sperando a tinta preta que nao chegou
    📦 informatica_acessorios

[4] ❌ | 확률: 0.69 | 차이: 0.11
    실제: Very Pos     → 예측: Pos
    📝 Boa qualidade do material e da impressão. Demora na entrega.
    📦 moveis_decoracao

[5] ❌ | 확률: 0.93 | 차이: 0.46
    실제: Very Pos     → 예측: Pos
    📝 Só nao entendi porque a caixa veio aberta. Mais fora isso tá perfeito veio tudo certinho.
    📦 beleza_saude

[6] ❌ | 확률: 0.01 | 차이: 0.05
    실제: Very Pos     → 예측: Pos
    📝 Vcs estao de parabens
    📦 relogios_presentes

[7] ❌ | 확률: 1.67 | 차이: 0.39
    실제: Very Pos     → 예측: Pos
    📝 Ok, produto de qualidade recomendo.
    📦 ferramentas_jardim

[8] ❌ | 확률: 0.48 | 차이: 0.01
    실제: Very Pos     → 예측: Pos
    📝 PRODOTO CONFORME DESCRITO.
    📦 eletronicos

[9] ❌ | 확률: 0.70 | 차이: 0.27
    실제: Very Pos     → 예측: Pos
    📝 Conforme vendido eu o recebi
    📦 brinquedos

[10] ❌ | 확률: 0.41 | 차이: 0.04
    실제: Very Pos     → 예측: Pos
    📝 L
    📦 utilidades_domesticas

======================================================================
🔍 Very Neg → Neg 오류 (262건) (총 244개 중 10개)
======================================================================

[1] ❌ | 확률: 1.01 | 차이: 0.16
    실제: Very Neg     → 예측: Neg
    📝 Recebi o relógio a tarde, mais precisamente às 18:05 do mesmo dia os ponteiros do análogico pararam de funcionar. Não estou usando preciso de uma solu
    📦 relogios_presentes

[2] ❌ | 확률: 1.46 | 차이: 0.16
    실제: Very Neg     → 예측: Neg
    📝 O tecido não era oq eu esperava, parece uma borracha, igual aos plásticos q forravamos as mesas em tempo do colégio. Abri apenas uma e a outra nem abr
    📦 pet_shop

[3] ❌ | 확률: 1.65 | 차이: 0.04
    실제: Very Neg     → 예측: Neg
    📝 não gostei do produto. muito fraco. não valeu a pena. você tira o produto da embalagem e no outro dia não tem mais o cheiro
    📦 cama_mesa_banho

[4] ❌ | 확률: 1.34 | 차이: 0.58
    실제: Very Neg     → 예측: Neg
    📝 Eu comprei dois golfinhos massageador. Porém fizeram duas postagens sendo que poderia ser feito em somente uma. Tive que buscar o pacote na agência do
    📦 beleza_saude

[5] ❌ | 확률: 1.67 | 차이: 0.48
    실제: Very Neg     → 예측: Neg
    📝 Produto não corresponde ao da foto pois o que fora enviado é de cor diversa e de qualidade inferior. Em supermercados locais o produto é vendido a men
    📦 telefonia_fixa

[6] ❌ | 확률: 0.82 | 차이: 0.06
    실제: Very Neg     → 예측: Neg
    📝 Efetuei a compra de duas cortinas, mas recebi somente uma.
    📦 cama_mesa_banho

[7] ❌ | 확률: 1.62 | 차이: 0.29
    실제: Very Neg     → 예측: Neg
    📝 Comprei um chaveador RCA que veio com defeito no cabo RCA e em uma das saídas de áudio. A lannister pediu uns dias para resolver o problema mas nada f
    📦 eletronicos

[8] ❌ | 확률: 1.62 | 차이: 0.90
    실제: Very Neg     → 예측: Neg
    📝 O produto veio diferente da q mostra na foto, não é de luxo, nem acolchoado como dizia.
    📦 instrumentos_musicais

[9] ❌ | 확률: 1.33 | 차이: 0.71
    실제: Very Neg     → 예측: Neg
    📝 Produto totalmente diferente do da foto não gostei pois falava que era corta luz plástico super fino
    📦 moveis_decoracao

[10] ❌ | 확률: 1.50 | 차이: 0.35
    실제: Very Neg     → 예측: Neg
    📝 O produto recebido não e o mesmo das fotos do anuncio.
    📦 automotivo

======================================================================
📊 검토 대상 요약
======================================================================
재라벨링 후보                1338개   ⭐⭐⭐ 최우선
전처리 개선                  376개   ⭐⭐ 중요
규칙 보완                    32개   ⭐ 낮음
Very Pos → Pos          209개   ⭐⭐⭐ 최우선
Very Neg → Neg          244개   ⭐⭐ 중요

💡 검토 순서: Very Pos↔Pos → 재라벨링 후보 → 전처리 개선
