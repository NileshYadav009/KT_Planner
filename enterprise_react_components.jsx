/**
 * ENTERPRISE KT PLANNER - REACT COMPONENTS
 * Production-ready React components for drag-drop mapping UI
 */

import React, { useEffect, useState, useRef } from 'react';

// =====================================================
// SENTENCE CARD COMPONENT
// =====================================================

export const SentenceCard = ({
  sentence,
  onMarkConfusing,
  onAddCodeRef,
  onEdit,
  isDragging = false
}) => {
  const confidenceColor = 
    sentence.confidence > 0.8 ? 'green' :
    sentence.confidence > 0.6 ? 'yellow' :
    'red';

  const statusBadgeColor = {
    'unmapped': 'gray',
    'mapped': 'blue',
    'needs_review': 'orange'
  }[sentence.status] || 'gray';

  return (
    <div
      className="sentence-card"
      draggable
      onDragStart={(e) => {
        e.dataTransfer?.setData('sentenceId', sentence.id);
        e.dataTransfer && (e.dataTransfer.effectAllowed = 'move');
      }}
      style={{
        padding: '12px',
        marginBottom: '8px',
        backgroundColor: sentence.is_confusing ? '#fff5e6' : '#f9f9f9',
        borderLeft: `4px solid ${confidenceColor}`,
        borderRadius: '4px',
        cursor: 'grab',
        boxShadow: isDragging ? '0 4px 12px rgba(0,0,0,0.15)' : 'none',
        transition: 'all 0.2s ease'
      }}
    >
      <p style={{ margin: '0 0 8px 0', fontSize: '14px', lineHeight: '1.4' }}>
        {sentence.text}
      </p>

      <div style={{ display: 'flex', gap: '8px', marginBottom: '8px', flexWrap: 'wrap' }}>
        <span style={{
          fontSize: '11px',
          padding: '2px 6px',
          backgroundColor: confidenceColor,
          color: 'white',
          borderRadius: '3px'
        }}>
          {(sentence.confidence * 100).toFixed(0)}%
        </span>

        <span style={{
          fontSize: '11px',
          padding: '2px 6px',
          backgroundColor: statusBadgeColor,
          color: 'white',
          borderRadius: '3px'
        }}>
          {sentence.status}
        </span>

        {sentence.is_confusing && (
          <span style={{
            fontSize: '11px',
            padding: '2px 6px',
            backgroundColor: '#ff9800',
            color: 'white',
            borderRadius: '3px'
          }}>
            ⚠️ Confusing
          </span>
        )}

        {sentence.has_code_ref && (
          <span style={{
            fontSize: '11px',
            padding: '2px 6px',
            backgroundColor: '#9c27b0',
            color: 'white',
            borderRadius: '3px'
          }}>
            📄 Code Ref
          </span>
        )}

        {sentence.is_important && (
          <span style={{
            fontSize: '11px',
            padding: '2px 6px',
            backgroundColor: '#f44336',
            color: 'white',
            borderRadius: '3px'
          }}>
            ⭐ Important
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: '6px', justifyContent: 'flex-end' }}>
        <button
          onClick={onEdit}
          style={{
            padding: '4px 8px',
            fontSize: '11px',
            backgroundColor: '#e3f2fd',
            border: '1px solid #2196f3',
            borderRadius: '3px',
            cursor: 'pointer',
            color: '#2196f3'
          }}
        >
          ✏️ Edit
        </button>

        <button
          onClick={onAddCodeRef}
          style={{
            padding: '4px 8px',
            fontSize: '11px',
            backgroundColor: '#f3e5f5',
            border: '1px solid #9c27b0',
            borderRadius: '3px',
            cursor: 'pointer',
            color: '#9c27b0'
          }}
        >
          📄 Code
        </button>

        <button
          onClick={onMarkConfusing}
          style={{
            padding: '4px 8px',
            fontSize: '11px',
            backgroundColor: '#fff3e0',
            border: '1px solid #ff9800',
            borderRadius: '3px',
            cursor: 'pointer',
            color: '#ff9800'
          }}
        >
          ⚠️ Confusing
        </button>
      </div>
    </div>
  );
};

// =====================================================
// SECTION COLUMN COMPONENT
// =====================================================

export const SectionColumn = ({
  section,
  sentences,
  onDropSentence,
  isDropTarget = false,
  totalMapped = 0,
  totalRequired = 0
}) => {
  const [isDragOver, setIsDragOver] = useState(false);
  const isCovered = totalMapped >= (section.required ? 2 : 1);

  return (
    <div
      className="section-column"
      onDragOver={(e) => {
        e.preventDefault();
        e.dataTransfer && (e.dataTransfer.dropEffect = 'move');
        setIsDragOver(true);
      }}
      onDragLeave={() => setIsDragOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setIsDragOver(false);
        const sentenceId = e.dataTransfer?.getData('sentenceId');
        if (sentenceId) {
          onDropSentence(sentenceId);
        }
      }}
      style={{
        flex: 1,
        minWidth: '250px',
        padding: '12px',
        backgroundColor: isDragOver ? '#e8f5e9' : '#f5f5f5',
        borderRadius: '6px',
        border: isDragOver ? '2px dashed #4caf50' : `2px solid ${isCovered ? '#4caf50' : '#ccc'}`,
        transition: 'all 0.2s ease',
        minHeight: '400px',
        overflowY: 'auto'
      }}
    >
      <div style={{ marginBottom: '12px' }}>
        <h3 style={{ margin: '0 0 4px 0', fontSize: '16px', color: '#333' }}>
          {section.title}
          {section.required && <span style={{ color: '#f44336' }}> *</span>}
        </h3>
        <p style={{ margin: '0', fontSize: '12px', color: '#999' }}>
          {sentences.length} mapped {isCovered ? '✓' : '✗'}
        </p>
        {section.description && (
          <p style={{ margin: '4px 0 0 0', fontSize: '12px', color: '#666', fontStyle: 'italic' }}>
            {section.description}
          </p>
        )}
      </div>

      <div>
        {sentences.length > 0 ? (
          sentences.map((sentence) => (
            <SentenceCard
              key={sentence.id}
              sentence={sentence}
              onMapToSection={() => {}}
              onMarkConfusing={() => {}}
              onAddCodeRef={() => {}}
              onEdit={() => {}}
            />
          ))
        ) : (
          <div style={{
            textAlign: 'center',
            color: '#999',
            padding: '20px',
            fontSize: '14px'
          }}>
            {isDragOver ? '📍 Drop sentences here' : '🔲 No sentences yet'}
          </div>
        )}
      </div>
    </div>
  );
};

// =====================================================
// DRAG & DROP CONTAINER
// =====================================================

export const DragDropContainer = ({
  sentences,
  sections,
  onSentenceMapped,
  projectId
}) => {
  const sentencesBySection = (sectionId) => {
    return sentences.filter((s) => s.assigned_section === sectionId);
  };

  const unassignedSentences = sentences.filter((s) => !s.assigned_section);

  return (
    <div
      className="drag-drop-container"
      style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}
    >
      {unassignedSentences.length > 0 && (
        <div
          className="unassigned-panel"
          style={{
            padding: '12px',
            backgroundColor: '#fce4ec',
            borderRadius: '6px',
            border: '2px solid #f06292'
          }}
        >
          <h3 style={{ margin: '0 0 8px 0', fontSize: '16px', color: '#c2185b' }}>
            📌 Unassigned Sentences ({unassignedSentences.length})
          </h3>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
            {unassignedSentences.map((sentence) => (
              <div
                key={sentence.id}
                draggable
                onDragStart={(e) => {
                  e.dataTransfer?.setData('sentenceId', sentence.id);
                }}
                style={{
                  padding: '8px 12px',
                  backgroundColor: 'white',
                  borderRadius: '4px',
                  cursor: 'grab',
                  border: '1px solid #f06292',
                  fontSize: '12px',
                  maxWidth: '200px',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap'
                }}
              >
                {sentence.text.substring(0, 50)}...
              </div>
            ))}
          </div>
        </div>
      )}

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
          gap: '16px'
        }}
      >
        {sections.map((section) => (
          <SectionColumn
            key={section.id}
            section={section}
            sentences={sentencesBySection(section.id)}
            onDropSentence={(sentenceId) => {
              onSentenceMapped(sentenceId, section.id);
            }}
            isDropTarget={true}
            totalMapped={sentencesBySection(section.id).length}
          />
        ))}
      </div>
    </div>
  );
};

// =====================================================
// WEBSOCKET UPDATES HOOK
// =====================================================

export const useWebSocketUpdates = (projectId, onUpdate) => {
  const wsRef = useRef(null);

  useEffect(() => {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    const wsUrl = `${protocol}//${window.location.host}/api/v1/ws/${projectId}`;
    
    wsRef.current = new WebSocket(wsUrl);

    wsRef.current.onopen = () => {
      console.log('WebSocket connected to', projectId);
    };

    wsRef.current.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('WebSocket message:', data);
        onUpdate(data);
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    };

    wsRef.current.onerror = (error) => {
      console.error('WebSocket error:', error);
    };

    return () => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.close();
      }
    };
  }, [projectId, onUpdate]);

  return wsRef.current;
};

// =====================================================
// INTEGRATION GUIDE
// =====================================================

/**
 * HOW TO USE THESE COMPONENTS:
 *
 * 1. Install dependencies:
 *    npm install zustand @dnd-kit/core @dnd-kit/utilities
 *
 * 2. Import in your React app:
 *    import { DragDropContainer, useWebSocketUpdates } from './enterprise_react_components'
 *
 * 3. Use in a component:
 *    <DragDropContainer
 *      sentences={sentences}
 *      sections={sections}
 *      onSentenceMapped={async (sentenceId, sectionId) => {
 *        await fetch('/api/v1/sentences/map', {
 *          method: 'POST',
 *          headers: { 'Content-Type': 'application/json' },
 *          body: JSON.stringify({
 *            sentence_id: sentenceId,
 *            section_id: sectionId,
 *            project_id: projectId
 *          })
 *        })
 *      }}
 *      projectId={projectId}
 *    />
 *
 * 4. Listen for real-time updates:
 *    useWebSocketUpdates(projectId, (update) => {
 *      if (update.event === 'sentence_mapped') {
 *        // Re-fetch or update local state
 *      }
 *    })
 */
