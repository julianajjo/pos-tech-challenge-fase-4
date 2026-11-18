# Configuração dos serviços Azure

O projeto usa dois recursos do **Azure AI Services**. Os dois têm plano gratuito (F0), suficiente para a demonstração. Contas de estudante podem usar o [Azure for Students](https://azure.microsoft.com/pt-br/free/students/), que não pede cartão de crédito.

## 1. Azure AI Speech (Speech to Text)

1. Acesse o [portal do Azure](https://portal.azure.com) e clique em **Criar um recurso**.
2. Pesquise **Speech** (Fala) e clique em **Criar**.
3. Preencha:
   - **Grupo de recursos**: crie um novo, por exemplo `rg-tech-challenge-4`;
   - **Região**: `Brazil South` (ou `East US`);
   - **Nome**: por exemplo `speech-tech-challenge-4`;
   - **Tipo de preço**: `Free F0`.
4. Clique em **Revisar + criar** e depois em **Criar**.
5. Abra o recurso criado e vá em **Chaves e Ponto de Extremidade**. Copie a **Chave 1** e a **Localização/Região** (ex.: `brazilsouth`).

## 2. Azure AI Language (Text Analytics)

1. Em **Criar um recurso**, pesquise **Language service** (Serviço de Linguagem) e clique em **Criar**.
2. Na tela de recursos adicionais, clique em **Continuar para criar seu recurso** (o *Text Analytics for Health* já vem incluído).
3. Use o mesmo grupo de recursos e região, um nome como `language-tech-challenge-4` e o tipo de preço `Free F0`.
4. Marque a confirmação de uso responsável de IA e crie o recurso.
5. Em **Chaves e Ponto de Extremidade**, copie a **Chave 1** e o **Ponto de extremidade** (ex.: `https://language-tech-challenge-4.cognitiveservices.azure.com/`).

## 3. Configurar o projeto

Na raiz do repositório:

```bash
cp .env.example .env
```

Edite o `.env`:

```
AZURE_SPEECH_KEY=<chave do Speech>
AZURE_SPEECH_REGION=brazilsouth
AZURE_LANGUAGE_KEY=<chave do Language>
AZURE_LANGUAGE_ENDPOINT=https://<seu-recurso>.cognitiveservices.azure.com/
```

O arquivo `.env` está no `.gitignore` e **nunca deve ser enviado ao GitHub**.

No Google Colab, as chaves podem ser cadastradas em **Secrets** (ícone de chave na barra lateral) e carregadas no notebook.

## 4. Testar

```bash
python scripts/run_audio.py data/samples/consulta_P001.wav --paciente P001
```

A saída deve mostrar `Transcrição (azure)` e, se o Text Analytics for Health estiver disponível na região, a lista de entidades de saúde reconhecidas.

## 5. Webhook do Microsoft Teams (opcional)

Para receber os alertas de severidade ALTA e CRÍTICA em um canal do Teams:

1. No canal, abra **Fluxos de trabalho** (Workflows) e escolha o modelo **Postar em um canal quando uma solicitação de webhook for recebida**.
2. Copie a URL gerada e coloque em `ALERT_WEBHOOK_URL` no `.env`.

O mesmo campo aceita webhooks do Slack.

## Custos e limites do plano gratuito

| Serviço | Limite mensal F0 |
|---|---|
| Speech to Text | 5 horas de áudio |
| Language (sentimento, entidades, frases-chave) | 5.000 registros de texto |
| Text Analytics for Health | 5.000 registros de texto |

Os valores podem mudar; confira a página de preços de cada serviço.
