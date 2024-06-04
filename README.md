# Introdução

Projeto feito para desenvolver redes neurais capazes de analisar corretamente uma porta XOR. Foi desenvolvido dois modelos de rede neural: `neuron.py` e `torch_net.py`. A diferença de cada um é o jeito que foi construído: o `torch_net.py` foi construído utilizando a biblioteca PyTorch, enquanto o `neuron.py` foi desenvolvido apenas com o NumPy. Note que os dois códigos apresentam um esquema de treino que ocorrerá 10 vezes e escolhe o melhor modelo. Isso ocorre devido à aleatoriedade dada aos pesos iniciais, podendo levar a um mínimo local. Além disso, foi escolhido que o número de neurônios escondidos seria de 2, pois assim o modelo identifica portas XOR com mais facilidade.

## Modelo 1 - `neuron.py`

Esse é o modelo construído apenas com NumPy. Ele utiliza os cálculos da propagação frontal para aderir um valor ao output e compara o valor estimado com o valor real do output. Assim, ele utiliza da fórmula para calcular a função de loss e depois começa o trabalho do backward propagation, seguindo a fórmula. Após ser calculada, re-atribui valores aos pesos, juntamente ao learning rate.

[Vídeo de demostração](https://youtu.be/fI8aRzsdEok)

## Modelo 2 - `torch_net.py`

Esse é o modelo construído utilizando PyTorch. Ele se aproveita das funcionalidades dessa biblioteca para criar uma rede neural com uma camada escondida de 2 neurônios e uma camada de saída. A propagação frontal (forward pass) é definida através de funções de ativação Sigmoid aplicadas nas camadas escondida e de saída. O treinamento utiliza a função de perda binária de entropia cruzada (`BCELoss`) e o otimizador Stochastic Gradient Descent (`SGD`).

[Vídeo de demonstração](https://youtu.be/N2qxRVeZ588)

### Como rodar

Para rodar o projeto, siga os seguintes passos:

1. Clone o repositório:
    ```bash
    git clone https://github.com/MarioVenturaMedeiros/Neural-XOR.git
    cd Neural-XOR
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Instale os requisitos:
    ```bash
    pip install -r requirements.txt
    ```

4. Execute o script `neuron.py`:
    ```bash
    python neuron.py
    ```

5. Execute o script `torch_net.py`:
    ```bash
    python torch_net.py
    ```

Note que os comandos acima estão assumindo que o usuário está usando Linux. Caso esteja usando Windows, substitua `python3` por `python` e `venv/bin/activate` por `venv\Scripts\activate`.