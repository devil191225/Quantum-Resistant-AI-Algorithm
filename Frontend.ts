// Frontend Architecture - React with TypeScript

// Main Components Structure:
// - App
//   - Authentication (Login/Registration)
//   - Dashboard
//     - PatientList
//     - PatientAdd
//   - PatientSession
//     - ConsentForm
//     - TranscriptionInterface
//       - LiveRecording
//       - FileUpload
//     - NoteEditor
//       - FormatSelector
//       - AIGeneratedContent
//       - ManualEditControls
//   - Settings
//     - UserPreferences
//     - NoteTemplates

// Example React component for the Dashboard

import React, { useState, useEffect } from 'react';
import { PatientList } from './components/PatientList';
import { PatientAdd } from './components/PatientAdd';
import { useAuth } from './hooks/useAuth';
import { fetchPatients } from './api/patientService';

const Dashboard = () => {
  const [patients, setPatients] = useState([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState(null);
  const [showAddPatient, setShowAddPatient] = useState(false);
  const { user } = useAuth();

  useEffect(() => {
    const loadPatients = async () => {
      try {
        setIsLoading(true);
        const data = await fetchPatients(user.id);
        setPatients(data);
      } catch (err) {
        setError('Failed to load patients. Please try again.');
        console.error('Error loading patients:', err);
      } finally {
        setIsLoading(false);
      }
    };

    loadPatients();
  }, [user.id]);

  const handleAddPatient = () => {
    setShowAddPatient(true);
  };

  const handlePatientAdded = (newPatient) => {
    setPatients([...patients, newPatient]);
    setShowAddPatient(false);
  };

  if (isLoading) return <div>Loading patients...</div>;
  if (error) return <div className="error-message">{error}</div>;

  return (
    <div className="dashboard-container">
      <header className="dashboard-header">
        <h1>Health Wise</h1>
        <h2>Patient Dashboard</h2>
        <button 
          className="primary-button add-patient-button" 
          onClick={handleAddPatient}
        >
          Add New Patient
        </button>
      </header>

      {showAddPatient ? (
        <PatientAdd onPatientAdded={handlePatientAdded} onCancel={() => setShowAddPatient(false)} />
      ) : (
        <PatientList patients={patients} />
      )}
    </div>
  );
};

export default Dashboard;

// Example TranscriptionInterface component

import React, { useState, useRef } from 'react';
import { uploadTranscript, startLiveTranscription } from '../api/transcriptionService';

const TranscriptionInterface = ({ patientId, onTranscriptionComplete }) => {
  const [isRecording, setIsRecording] = useState(false);
  const [transcript, setTranscript] = useState('');
  const [uploadFile, setUploadFile] = useState(null);
  const [processingStatus, setProcessingStatus] = useState(null);
  const recorderRef = useRef(null);
  const mediaStreamRef = useRef(null);

  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setUploadFile(file);
    }
  };

  const handleFileUpload = async () => {
    if (!uploadFile) return;
    
    try {
      setProcessingStatus('Uploading and processing transcript...');
      const result = await uploadTranscript(patientId, uploadFile);
      onTranscriptionComplete(result.transcription);
      setProcessingStatus('Transcript processed successfully');
    } catch (error) {
      setProcessingStatus('Error processing transcript');
      console.error('Upload error:', error);
    }
  };

  const startRecording = async () => {
    try {
      setProcessingStatus('Initializing recording...');
      
      // Request microphone access
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      mediaStreamRef.current = stream;
      
      // Initialize recorder
      const options = { mimeType: 'audio/webm' };
      const mediaRecorder = new MediaRecorder(stream, options);
      recorderRef.current = mediaRecorder;
      
      // Set up recording data
      const audioChunks = [];
      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunks.push(event.data);
        }
      };
      
      // Handle recording stop
      mediaRecorder.onstop = async () => {
        const audioBlob = new Blob(audioChunks, { type: 'audio/webm' });
        setProcessingStatus('Processing recording...');
        
        try {
          // Send for transcription
          const result = await startLiveTranscription(patientId, audioBlob);
          onTranscriptionComplete(result.transcription);
          setProcessingStatus('Transcription complete');
        } catch (error) {
          setProcessingStatus('Error processing recording');
          console.error('Transcription error:', error);
        }
      };
      
      // Start recording
      mediaRecorder.start();
      setIsRecording(true);
      setProcessingStatus('Recording in progress...');
    } catch (error) {
      setProcessingStatus('Error accessing microphone');
      console.error('Recording error:', error);
    }
  };

  const stopRecording = () => {
    if (recorderRef.current && isRecording) {
      recorderRef.current.stop();
      setIsRecording(false);
      
      // Stop and release media stream
      if (mediaStreamRef.current) {
        mediaStreamRef.current.getTracks().forEach(track => track.stop());
      }
    }
  };

  return (
    <div className="transcription-interface">
      <h3>Session Recording</h3>
      
      <div className="transcription-options">
        <div className="live-recording-section">
          <h4>Live Recording</h4>
          <p>Record the conversation in real-time</p>
          
          {!isRecording ? (
            <button 
              className="primary-button"
              onClick={startRecording}
            >
              Start Recording
            </button>
          ) : (
            <button 
              className="danger-button"
              onClick={stopRecording}
            >
              Stop Recording
            </button>
          )}
        </div>
        
        <div className="upload-section">
          <h4>Upload Transcript</h4>
          <p>Upload a pre-recorded audio file</p>
          
          <input 
            type="file" 
            accept="audio/*" 
            onChange={handleFileChange} 
          />
          
          <button 
            className="secondary-button"
            onClick={handleFileUpload}
            disabled={!uploadFile}
          >
            Process File
          </button>
        </div>
      </div>
      
      {processingStatus && (
        <div className="processing-status">
          {processingStatus}
        </div>
      )}
    </div>
  );
};

export default TranscriptionInterface;

// Note Editor Component for reviewing and editing AI-generated notes

import React, { useState, useEffect } from 'react';
import { saveNote } from '../api/noteService';

const NOTE_FORMATS = [
  { id: 'soap', name: 'SOAP Note' },
  { id: 'clinical', name: 'Clinical Note' },
  { id: 'progress', name: 'Progress Note' },
  { id: 'consultation', name: 'Consultation Note' }
];

const NoteEditor = ({ patientId, transcription, initialFormat = 'clinical' }) => {
  const [noteFormat, setNoteFormat] = useState(initialFormat);
  const [generatedNote, setGeneratedNote] = useState(null);
  const [editedNote, setEditedNote] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [saveStatus, setSaveStatus] = useState(null);

  useEffect(() => {
    const generateNote = async () => {
      if (!transcription) return;
      
      setIsLoading(true);
      try {
        // Call AI service to generate structured note
        const result = await fetchGeneratedNote(patientId, transcription, noteFormat);
        setGeneratedNote(result);
        setEditedNote(result); // Initialize edited note with the generated one
      } catch (error) {
        console.error('Error generating note:', error);
      } finally {
        setIsLoading(false);
      }
    };

    generateNote();
  }, [patientId, transcription, noteFormat]);

  const handleFormatChange = (e) => {
    const newFormat = e.target.value;
    setNoteFormat(newFormat);
  };

  const handleNoteEdit = (section, content) => {
    setEditedNote({
      ...editedNote,
      [section]: content
    });
  };

  const handleSaveNote = async () => {
    setSaveStatus('Saving note...');
    try {
      await saveNote(patientId, editedNote, noteFormat);
      setSaveStatus('Note saved successfully');
    } catch (error) {
      setSaveStatus('Error saving note');
      console.error('Save error:', error);
    }
  };

  if (isLoading) return <div>Generating clinical note...</div>;
  if (!generatedNote) return <div>No transcription data available</div>;

  return (
    <div className="note-editor">
      <h3>Clinical Note Editor</h3>
      
      <div className="format-selector">
        <label htmlFor="note-format">Note Format:</label>
        <select 
          id="note-format"
          value={noteFormat}
          onChange={handleFormatChange}
        >
          {NOTE_FORMATS.map(format => (
            <option key={format.id} value={format.id}>
              {format.name}
            </option>
          ))}
        </select>
      </div>
      
      <div className="note-sections">
        {Object.entries(editedNote).map(([section, content]) => (
          <div key={section} className="note-section">
            <h4>{formatSectionTitle(section)}</h4>
            <textarea
              value={content || ''}
              onChange={(e) => handleNoteEdit(section, e.target.value)}
              rows={4}
              placeholder={`Enter ${formatSectionTitle(section)} information`}
            />
          </div>
        ))}
      </div>
      
      <div className="note-actions">
        <button 
          className="primary-button"
          onClick={handleSaveNote}
        >
          Save Note
        </button>
        
        <button 
          className="secondary-button"
          onClick={() => setEditedNote(generatedNote)}
        >
          Reset to AI Generated
        </button>
      </div>
      
      {saveStatus && (
        <div className="save-status">
          {saveStatus}
        </div>
      )}
    </div>
  );
};

// Helper function to format section titles
const formatSectionTitle = (key) => {
  return key
    .split('_')
    .map(word => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');
};

export default NoteEditor;