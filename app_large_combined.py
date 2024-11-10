import streamlit as st
import tensorflow as tf
import pickle
import keras_nlp
import re

MAX_SEQUENCE_LENGTH = 40

# Load the model
loaded_model = tf.keras.models.load_model("transformer_model_large_combined")

# Load jw_vocab
with open("jw_vocab_large_combined.pkl", "rb") as f:
    jw_vocab = pickle.load(f)

# Load rm_vocab
with open("rm_vocab_large_combined.pkl", "rb") as f:
    rm_vocab = pickle.load(f)

# Load tokenizers
jw_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=jw_vocab, lowercase=False
)
rm_tokenizer = keras_nlp.tokenizers.WordPieceTokenizer(
    vocabulary=rm_vocab, lowercase=False
)


# Function to check if the text contains only Jawi characters
def is_jawi_text(text):
    jawi_pattern = re.compile(r"^[\u0600-\u06FF\s]+$")
    return bool(jawi_pattern.match(text))


def translate_text(input_text):
    if not is_jawi_text(input_text):
        st.warning("Sila masukkan teks Jawi sahaja.")
        return ""

    # Tokenize the input sentence
    input_tokens = jw_tokenizer([input_text]).to_tensor(
        shape=(None, MAX_SEQUENCE_LENGTH)
    )

    # Generate translation
    translated_tokens = keras_nlp.samplers.GreedySampler()(
        lambda prompt, cache, index: (
            loaded_model([input_tokens, prompt])[:, index - 1, :],
            None,
            cache,
        ),
        prompt=tf.concat(
            [
                tf.fill(
                    (tf.shape(input_tokens)[0], 1), rm_tokenizer.token_to_id("[START]")
                ),
                tf.fill(
                    (tf.shape(input_tokens)[0], MAX_SEQUENCE_LENGTH - 1),
                    rm_tokenizer.token_to_id("[PAD]"),
                ),
            ],
            axis=-1,
        ),
        end_token_id=rm_tokenizer.token_to_id("[END]"),
        index=1,
    )

    # Detokenize the translated tokens
    translated_text = rm_tokenizer.detokenize(translated_tokens)

    return (
        translated_text.numpy()[0]
        .decode("utf-8")
        .replace("[PAD]", "")
        .replace("[START]", "")
        .replace("[END]", "")
        .strip()
    )


# Add your logo
logo_url = "https://static.wixstatic.com/media/a82f81_ee5d3c423b80433eaac59bb71b2bba16~mv2.png"  # Replace with the URL of your logo
st.image(logo_url, width=100)  # Adjust width as needed

# Streamlit app
st.title("Deep Learning for Classical Malay Text Transliteration")

# Input text box
input_text = st.text_input("Masukkan Jawi Teks:")

# Translate button
if st.button("Transliterasi"):
    if input_text:
        translation = translate_text(input_text)
        if translation:
            st.success(f"Hasil Transliterasi:  {translation}")
    else:
        st.warning("Sila masukkan teks untuk transliterasi.")

st.subheader("Contoh Teks")
st.write("ادوهاي")
st.write("بركروت داهيڽ")
st.write("امڤون توانكو ڤوتري")
st.write("دي سوده ترچدرا تروق")
st.write("مات راتو سيتي بانون ممبولت")
st.write("لقسامان كستوري كيت تيدق اكن داڤت برتاهن لبيه لاما جک برتروسن بڬيني")
