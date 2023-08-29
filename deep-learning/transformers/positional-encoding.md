# Positional Encoding

Lembrando das RNNs, havia (de modo inerente ao modelo) uma ideia de sequencialidade entre os inputs (palavras).

Isso porque, tanto no encoder (vetor de anotação, vetor de contexto, hidden state) quanto no decoder (palavra, prevista, hidden state) as informações obtidas sobre a palavra $i$ levavam em consideração informações sobre as palavras anteriores.

Até mesmo pelo fato do input ser sequencial.

\tipbox{Questão sobre paralelismo em RNNs e Transformers}

Agora, no caso de Transformers, essa ideia de sequencialidade nas palavras de input não são inerentes à arquitetura do modelo. $\rightarrow$ Relita sobre como o mecanismo de Self-Attention funciona sem ideia de sequencialidade.

Apesar disso, é evidente que a posição das palavras em uma frase tem muita importância na interpretação do seu significado naquela frase.

Assim, Positional Encoding é uma maneira de embutir essas informações sobre localidade no embedding das palavras.

Esse será o embedding que é forncecido como $v_i$ para o encoder.

De maneira geral, somamos ao embedding de uma palavra um vetor que tenha a informação sobre a posição da palavra na frase. 

\section{Entendendo Word embeddings}

\section{Embedding contextualizado vs embedding tradicional de palavras}
Como vamos do embedding contextualizado entregue pelo encoder de Transformer para uma frase em outro idioma (no contexto de tradução).

A partir de uma perspectiva de muito alto nível, o encoder e o decoder interagem da seguinte maneira:

\begin{center}
    \input{fig4}
\end{center}


\qbox{Novas matrizes K e V?}

Aqui, podemos pensar primeiramente sobre o mecanismo de masked Self-Attention.

\qbox{Considerando que Masked Self Attention faz com que o modelo não roube durante o treinamento, como esse mecanismo atua durante inferência?}

\tipbox{Input do decoder é diferente em treinamento e inferência:
\begin{itemize}
    \item Treinamento: frase inteira $\rightarrow$ paralelização.
    \item Inferência: uma palavra por vez. Inferência precisa da natureza sequencial.
\end{itemize}}

\qbox{Em treinamento a frase predita \textbf{y} é predita de uma só vez? (Todas as palavras simultaneamente?) 

Input diferente $\Rightarrow$ output diferente?

Acho que isso faz sentido. Talvez possamos inferir todas as próximas palavras simultaneamente.}

\subsubsection*{Masked Selft-Attention}
Na prática, quando estamos falando de Transformers Decoder, temos a seguinte figura esquemática:

\begin{center}
    \input{fig5.tex}
\end{center}

Aqui, o podemos diferenciar o funcionamento do decoder em estágio de treinamento e inferência. Primeiramente trataremos do período de treinamento.

\subsection{Decoder em Estágio de Treinamento}
Nessa situação, podemos pensar que o modelo recebe como inputs os seguintes dados:
\begin{enumerate}
    \item Frase inteiramente traduzida
    \item embeddings contextualizados das palavras no idioma original.
\end{enumerate}

Em um primeiro momento de atuação do Decoder, a frase inteiramente traduzida é submetida à Masked Self Attention.

Aqui, a ideia é estabelecer a relação do próximo output (próxima palavra predita) com os outputs anteriores.

\tipbox{Podemos pensar em uma analogia com os hidden states do decoder de uma RNN}

Para entender como Masked Self Attention funciona, podemos pensar na matriz de Self-Attention:

Dado as palavras: $t_1, t_2  \text{ e } t_3$ ($t$ de ``traduzido''), temos a seguinte matriz:

\qbox{Como os Transformers lidam com a geração de frases traduzidas que tenham comprimento diferente do da frase no idioma original?}
\tipbox{Isso depende de quando o $<$EOS$>$ será predito. Podemos lembrar do funcionamento de RNNs.}


\begin{center}
\begin{tikzpicture}
\matrix (m) [matrix of nodes, style={nodes={rectangle,draw,minimum width=3em}}, minimum height=1.5em, row sep=-\pgflinewidth, column sep=-\pgflinewidth]
{
{} & {} & {} \\ % Row 1
{} & {} & {} \\ % Row 2
{} & {} & {} \\ % Row 3
};

% Labels for columns
\node[above=1mm of m-1-1] {$t_1$};
\node[above=1mm of m-1-2] {$t_2$};
\node[above=1mm of m-1-3] {$t_3$};

% Labels for rows
\node[left=1mm of m-1-1] {$t_1$};
\node[left=1mm of m-2-1] {$t_2$};
\node[left=1mm of m-3-1] {$t_3$};

\end{tikzpicture}
\end{center}

Onde o elemento $ij$ denota a intensidade da relação contextual entre as palavras $t_i$ e $t_j$.

Agora, perceba que, em momento de treinamento, como já possuímos a frase traduzida inteira ($t_1, t_2 \text{ e } t_3$), essa matriz pode ser inteiramente preenchida no momento inicial. Ou seja, podemos traçar as relações contextuais: $t_1t_1, t_1t_2, t_1t_3$ mesmo que, na prática, em momento de inferência, isso não seria possível (já que ainda não existiria $t_2$ ou $t_3$).

Nesse ponto, entra o conceito de Masked Self-Attention A ideia aqui é transformar a matriz de Sel-Attention (ilustrada anteriormente) em algo do tipo:

\begin{center}
\begin{tikzpicture}
\matrix (m) [matrix of nodes, style={nodes={rectangle,draw,minimum width=3em, text depth=0.5ex, text height=2ex}}, row sep=-\pgflinewidth, column sep=-\pgflinewidth]
{
{} & $-\infty$ & $-\infty$ \\ % Row 1
{} & {} & $-\infty$ \\ % Row 2
{} & {} & {} \\ % Row 3
};

% Labels for columns
\node[above=1mm of m-1-1] {$t_1$};
\node[above=1mm of m-1-2] {$t_2$};
\node[above=1mm of m-1-3] {$t_3$};

% Labels for rows
\node[left=1mm of m-1-1] {$t_1$};
\node[left=1mm of m-2-1] {$t_2$};
\node[left=1mm of m-3-1] {$t_3$};

\end{tikzpicture}
\end{center}

\tipbox{Após Softmax, cada $-\infty$ se torna 0}

Desse modo, perceba que, apesar da frase inteira estar disponível, a matriz de Self-Attention resultante estará simulando o que ocorre em tempo de inferência.

Ou seja, $t_1$ só pode ter relação contextual analisada com $t_1$. $\rightarrow$ Somente $t_1t_1$.

$t_2$ pode ter relação com $t_1$ e $t_2$. $\Rightarrow t_2t_1;t_2t_2$.

E assim por diante. Perceba que dessa maneira podemos dar a frase inteira como input simulando uma geração sequencial das palavras.

Esse estágio de Self-Attention no Decoder é muito importante pois a próxima palavra traduzida prevista não está somente em função do output do Encoder. Essa palavra deve estar em função de todas as palavras anteriormente traduzidas.

\subsubsection*{Encoder-Decoder Attention}
Nesse ponto, depois de passarmos por Masked-Self Attention, com o que estamos lidando?

Primeiro obtemos a matriz de self-attention e a utilizamos para obter o embedding contextualizado de uma palava traduzida em relação à própria frase traduzida.

Aqui, segundo um esquema de self-Attention muito semelhante ao ilustrado em (), no momento de treinamento, cada palavra da frase traduzida origina um vetor $y$:

\begin{center}
\input{fig6.tex}
\end{center}

Agora, chegamos à etapa de Attention entre Encoder e Decoder. Esse conceito de Attention é muito semelhante ao proposto por \cite{attention}, no qual quantificamos a relação contextual da próxima palavra predita com todas as palavras da frase no idioma original.

Perceba que se formos utilizar a analogia de Q, K, V nesse caso de Encoder-Decoder Attention, poderemos pensar em um esquema como o seguinte:

\begin{center}
    \input{fig7}
\end{center}

Desse modo, cada vetor contextualizado resultado do Encoder é relacionado com cada vetor contextualizado da palavra sendo predita com as palavras da frase original. Obtemos scores dessa relação. A palavra mais relacionada terá maior score e, então, influenciará mais o output.

Ao final de uma camada de Transformer's Decoder, teremos um output influenciado por todas as palavras (contextualizadas) da frase original e pelas palavras já traduzidas (contextualizadas).

\subsubsection*{Considerações Sobre Treinamento}
É razoável pensar tradução como um problema de classificação. Portanto, podemos usar Cross-Entropy como função de perda.

\tipbox{Em momento de treinamento, o erro de uma predição não é propagado para a predição das próximas palavras traduzidas.

Assim, de fato usamos como ``palavras previamente traduzidas'' as palavras corretamente traduzidas. Esse conceito é conhecido como ``teacher forcing''.

Isso faz com que o processo de treinamento de uma rede Transformer seja totalmente paralelizável.}

\qbox{O Encoder é treinado juntamente com o Decoder no contexto de tradução de texto?}
\tipbox{Acredito que, nessa abordagem de Transformer, como mostrado em \cite{self-attention}, Encoder e Decoder são trainados em conjuntom como uma mesma rede (que são)

Contudo, as arquiteturas Encoder/Decoder poderiam ser treinadas separadamente. Por exemplo as aquiteruras Encoder only (BERT) e Decoder only (GPT)}



\bibliographystyle{apalike}
\bibliography{bibliografia}


\end{document}

