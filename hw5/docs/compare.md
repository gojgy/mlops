# База и адаптеры: заголовки arXiv

Итоговый адаптер обучен на 3072 примерах, полная эпоха. Для сравнения показан предыдущий адаптер на 512 примерах. База — Qwen3-1.7B; генерация жадная, до 40 токенов.

## 1. arxiv:1203.0108

**Аннотация:** In the present paper, we consider the problem of matrix completion with noise. Unlike previous works, we consider quite general sampling distribution and we do not need to know or to estimate the variance of the noise. Two new nuclear-norm penalized estimators are proposed, one of them of "square-root" type. We analyse their performance under high-dimensional scaling and provide non-asymptotic bounds on the Frobenius norm error. Up to a logarithmic factor, these performance guarantees are minimax optimal in a number of circumstances.

**Авторский заголовок:** Noisy low-rank matrix completion with general sampling distribution

**База:**

> Matrix completion with noise: General sampling distribution and minimax-optimal performance

**Адаптер, 512 примеров:**

> Matrix Completion with Noise

**Итоговый адаптер, 3072 примера:**

> Matrix Completion with General Sampling

## 2. arxiv:0805.0779

**Аннотация:** Linear processes are defined as a discrete-time convolution between a kernel and an infinite sequence of i.i.d. random variables. We modify this convolution by introducing decimation, that is, by stretching time accordingly. We then establish central limit theorems for arrays of squares of such decimated processes. These theorems are used to obtain the asymptotic behavior of estimators of the spectral density at specific frequencies. Another application, treated elsewhere, concerns the estimation of the long-memory parameter in time-series, using wavelets.

**Авторский заголовок:** Central Limit Theorems for arrays of decimated linear processes

**База:**

> Asymptotic behavior of estimators of the spectral density for decimated linear processes

**Адаптер, 512 примеров:**

> Decimated linear processes and asymptotic behavior of estimators of the spectral density

**Итоговый адаптер, 3072 примера:**

> Decimated linear processes and spectral estimation

## 3. arxiv:0707.2116

**Аннотация:** In this paper, we develop an approach for the exact determination of the minimum sample size for the estimation of a Poisson parameter with prescribed margin of error and confidence level. The exact computation is made possible by reducing infinite many evaluations of coverage probability to finite many evaluations. Such reduction is based on our discovery that the minimum of coverage probability with respect to a Poisson parameter bounded in an interval is attained at a discrete set of finite many values.

**Авторский заголовок:** Exact Computation of Minimum Sample size for Estimation of Poisson Parameters

**База:**

> Exact determination of minimum sample size for Poisson parameter estimation with prescribed margin of error and confidence level

**Адаптер, 512 примеров:**

> Exact determination of the minimum sample size for Poisson parameter estimation

**Итоговый адаптер, 3072 примера:**

> Exact determination of the minimum sample size for Poisson parameter estimation

## 4. arxiv:1208.1969

**Аннотация:** An approach for engineering student exercises using the Internet is described. In this approach, for a given exercise, each student receives the same problem, but with different data. The exercise content can be static or dynamic, and the dynamic form can be timeless or real-time. The implementation provides immediate feedback to the students, letting them know if their submitted answers are correct. Student results for each exercise are recorded in log files which are available to the instructor. Example exercises from engineering computer security and cryptography courses are presented.

**Авторский заголовок:** An Internet Approach for Engineering Student Exercises

**База:**

> Engineering student exercises with dynamic data and real-time feedback

**Адаптер, 512 примеров:**

> Engineering student exercises using the Internet

**Итоговый адаптер, 3072 примера:**

> Engineering Student Exercises Using the Internet

## 5. arxiv:0910.3757

**Аннотация:** Sufficient conditions for global stabilization of nonlinear systems with delayed input by means of approximate predictors are presented. An approximate predictor is a mapping which approximates the exact values of the stabilizing input for the corresponding system with no delay. A systematic procedure for the construction of approximate predictors is provided for globally Lipschitz systems. The resulting stabilizing feedback can be implemented by means of a dynamic distributed delay feedback law. Illustrating examples show the efficiency of the proposed control strategy.

**Авторский заголовок:** Stabilization by Means of Approximate Predictors for Systems with Delayed Input

**База:**

> Sufficient conditions for global stabilization of nonlinear systems with delayed input using approximate predictors

**Адаптер, 512 примеров:**

> Approximate predictors for global stabilization of nonlinear systems with delayed input

**Итоговый адаптер, 3072 примера:**

> Approximate Predictors for Global Stabilization of Nonlinear Systems with Delay

## Проверка на 30 дополнительных статьях

Аннотации выбраны из test до получения ответов, seed 1337; в обучении и валидации не участвовали. Использованы короткие аннотации без TeX. ROUGE-L — F1 совпадения последовательности слов с авторским заголовком, нижний регистр, без стемминга. Это лексическое сходство, а не точность или оценка смысла.

| Модель | Средний ROUGE-L | Средняя длина, слов |
|---|---:|---:|
| База | 0,3059 | 12,1 |
| 512 примеров | 0,3874 | 8,5 |
| 3072 примера | 0,3679 | 8,1 |

Дополнительное обучение не дало устойчивого выигрыша. Есть улучшения: в `arxiv:1009.0896` новый заголовок вернул crossbar и аппаратную реализацию. Есть потери: в `arxiv:1208.1179` исчез risk-sensitive control, в `arxiv:1106.2338` ответ стал слишком общим. Итоговым выбран адаптер на 3072 примерах: его более компактные заголовки предпочтительны по форме. Val loss улучшился с 1,4040 до 1,3691, но ROUGE-L снизился, поэтому устойчивое улучшение точности не заявляется. Адаптер: `models/experiment3k/adapter_all_layers`; команда повторного сравнения — `make compare`.
