// mobile/expo-app/App.js
import React, { useState } from 'react';
import { StyleSheet, Text, View, Button, ActivityIndicator, FlatList, Linking } from 'react-native';
import * as DocumentPicker from 'expo-document-picker';
import Constants from 'expo-constants';

const BACKEND_URL = Constants.manifest?.extra?.BACKEND_URL || "http://10.0.2.2:8000"; // use emulator loopback for Android emulator

export default function App() {
  const [loading, setLoading] = useState(false);
  const [report, setReport] = useState(null);

  const pickAndUpload = async () => {
    const res = await DocumentPicker.getDocumentAsync({ type: "application/pdf" });
    if (res.type !== "success") return;

    setLoading(true);
    try {
      const uri = res.uri;
      const filename = res.name || "resume.pdf";
      const filetype = "application/pdf";

      const formData = new FormData();
      formData.append("file", {
        uri,
        name: filename,
        type: filetype,
      });

      const resp = await fetch(`${BACKEND_URL}/analyze`, {
        method: "POST",
        body: formData,
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });

      if (!resp.ok) {
        const err = await resp.json();
        alert("Analysis failed: " + JSON.stringify(err));
        setLoading(false);
        return;
      }
      const json = await resp.json();
      // attach full download url
      const reportUrl = BACKEND_URL + json.report_url;
      json.report_url_full = reportUrl;
      setReport(json);
    } catch (err) {
      alert("Upload error: " + err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <View style={styles.container}>
      <Text style={styles.title}>AI Resume Profiler</Text>
      <Button title="Pick & Upload Resume (PDF)" onPress={pickAndUpload} />
      {loading && <ActivityIndicator style={{ marginTop: 20 }} />}
      {report && (
        <View style={{ marginTop: 20, width: '90%' }}>
          <Text style={{ fontSize: 18, fontWeight: 'bold' }}>{report.candidate_name}</Text>
          <Text style={{ marginVertical: 6 }}>Overall Score: {report.weighted_score}/100</Text>
          <FlatList
            data={Object.entries(report.trait_scores)}
            keyExtractor={(item) => item[0]}
            renderItem={({ item }) => (
              <View style={styles.row}>
                <Text style={{ fontWeight: 'bold' }}>{item[0]}</Text>
                <Text>{item[1]}/100</Text>
              </View>
            )}
          />
          <Button title="Open PDF Report" onPress={() => Linking.openURL(report.report_url_full)} />
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, alignItems: 'center', paddingTop: 80 },
  title: { fontSize: 22, marginBottom: 20 },
  row: { flexDirection: 'row', justifyContent: 'space-between', paddingVertical: 6 }
});
