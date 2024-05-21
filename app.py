import streamlit as st
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# Fungsi untuk melakukan klastering
def perform_clustering(data, num_clusters):
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(data)
    return kmeans.labels_

# Fungsi untuk menampilkan visualisasi klaster
def plot_clusters(data, labels):
    sns.scatterplot(data.iloc[:, 0], data.iloc[:, 1], hue=labels, palette='viridis', legend='full')
    plt.xlabel('Fitur 1')
    plt.ylabel('Fitur 2')
    st.pyplot()

def main():
    st.title('Klasifikasi Sistem Kepuasan dengan Klastering')
    st.write('Upload file CSV dengan kolom: pendidikan, umur, q1, q2, q3, q4, q5, q6, q7, q8, q9, q10, q11.')

    uploaded_file = st.file_uploader("Upload file CSV", type=["csv"])
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        required_columns = ['pendidikan', 'umur', 'q1', 'q2', 'q3', 'q4', 'q5', 'q6', 'q7', 'q8', 'q9', 'q10', 'q11']

        if all(col in data.columns for col in required_columns):
            st.write("Data awal:")
            st.write(data)

            num_clusters = st.slider("Pilih jumlah klaster:", min_value=2, max_value=10, value=3, step=1)

            if st.button("Klastering"):
                # Hanya gunakan kolom yang diperlukan untuk klastering
                cluster_data = data[required_columns]
                labels = perform_clustering(cluster_data, num_clusters)
                st.write("Label klaster:")
                st.write(labels)

                # Tambahkan label klaster ke data
                data['Klaster'] = labels
                st.write("Data dengan label klaster:")
                st.write(data)

                # Visualisasi klaster
                plot_clusters(cluster_data, labels)
        else:
            st.write("File CSV tidak memiliki kolom yang diperlukan.")

if __name__ == "__main__":
    main()
