<template>
  <div id="app">
    <header class="header">
      <div class="container">
        <h1>Sentiment Analysis Chat</h1>
        <div class="connection-status">
          <span :class="['status-indicator', isConnected ? 'connected' : 'disconnected']"></span>
          {{ isConnected ? 'Connected' : 'Disconnected' }}
        </div>
      </div>
    </header>

    <main class="main">
      <div class="container">
        <div class="chat-container">
          <div class="messages" ref="messagesContainer">
            <div 
              v-for="message in messages" 
              :key="message.id"
              :class="['message', message.sender]"
            >
              <div class="message-content">
                <p>{{ message.text }}</p>
                <div v-if="message.sentiment" class="sentiment-info">
                  <div class="sentiment-badge" :class="message.sentiment.sentiment">
                    {{ message.sentiment.sentiment }}
                  </div>
                  <div class="confidence">
                    Confidence: {{ (message.sentiment.confidence * 100).toFixed(1) }}%
                  </div>
                </div>
              </div>
              <div class="message-time">
                {{ formatTime(message.timestamp) }}
              </div>
            </div>
            
            <div v-if="isTyping" class="message bot typing">
              <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
              </div>
            </div>
          </div>
        </div>

        <div class="input-container">
          <div class="input-group">
            <textarea
              v-model="inputText"
              @keydown="handleKeyDown"
              @input="handleInput"
              placeholder="Type your message here..."
              :disabled="!isConnected"
              rows="3"
              maxlength="1000"
            ></textarea>
            <div class="input-info">
              <span class="char-count">{{ inputText.length }}/1000</span>
              <button 
                @click="sendMessage"
                :disabled="!inputText.trim() || !isConnected"
                class="send-button"
              >
                <span v-if="!isSending">Send</span>
                <span v-else>Sending...</span>
              </button>
            </div>
          </div>
        </div>
      </div>
    </main>

    <footer class="footer">
      <div class="container">
        <div class="stats">
          <div class="stat">
            <span class="label">Messages:</span>
            <span class="value">{{ messageCount }}</span>
          </div>
          <div class="stat">
            <span class="label">Positive:</span>
            <span class="value positive">{{ sentimentCounts.positive }}</span>
          </div>
          <div class="stat">
            <span class="label">Negative:</span>
            <span class="value negative">{{ sentimentCounts.negative }}</span>
          </div>
          <div class="stat">
            <span class="label">Neutral:</span>
            <span class="value neutral">{{ sentimentCounts.neutral }}</span>
          </div>
        </div>
      </div>
    </footer>
  </div>
</template>

<script>
import { io } from 'socket.io-client'
import { nextTick } from 'vue'

export default {
  name: 'App',
  data() {
    return {
      socket: null,
      isConnected: false,
      inputText: '',
      messages: [],
      isTyping: false,
      isSending: false,
      messageId: 0
    }
  },
  computed: {
    messageCount() {
      return this.messages.filter(m => m.sender === 'user').length
    },
    sentimentCounts() {
      const counts = {
        positive: 0,
        negative: 0,
        neutral: 0
      }
      
      this.messages.forEach(message => {
        if (message.sentiment) {
          const sentiment = message.sentiment.sentiment
          if (counts[sentiment] !== undefined) {
            counts[sentiment]++
          }
        }
      })
      
      return counts
    }
  },
  mounted() {
    this.initializeSocket()
    this.loadStoredMessages()
  },
  beforeUnmount() {
    if (this.socket) {
      this.socket.disconnect()
    }
  },
  methods: {
    initializeSocket() {
      this.socket = io('http://localhost:5000')
      
      this.socket.on('connect', () => {
        this.isConnected = true
        console.log('Connected to server')
      })
      
      this.socket.on('disconnect', () => {
        this.isConnected = false
        console.log('Disconnected from server')
      })
      
      this.socket.on('sentiment_result', (data) => {
        this.handleSentimentResult(data)
      })
      
      this.socket.on('realtime_result', (data) => {
        this.handleRealtimeResult(data)
      })
      
      this.socket.on('error', (data) => {
        this.handleError(data)
      })
      
      this.socket.on('connected', (data) => {
        console.log('Server message:', data.message)
      })
    },
    
    sendMessage() {
      if (!this.inputText.trim() || !this.isConnected || this.isSending) {
        return
      }
      
      this.isSending = true
      
      const userMessage = {
        id: this.messageId++,
        sender: 'user',
        text: this.inputText.trim(),
        timestamp: new Date().toISOString()
      }
      
      this.messages.push(userMessage)
      this.saveMessages()
      
      // Enviar al servidor
      this.socket.emit('analyze_realtime', {
        text: this.inputText.trim()
      })
      
      this.inputText = ''
      this.isSending = false
      
      // Scroll al final
      this.scrollToBottom()
    },
    
    handleSentimentResult(data) {
      const botMessage = {
        id: this.messageId++,
        sender: 'bot',
        text: data.original_text,
        sentiment: {
          sentiment: data.sentiment,
          confidence: data.confidence,
          scores: data.scores
        },
        timestamp: data.timestamp
      }
      
      this.messages.push(botMessage)
      this.saveMessages()
      this.scrollToBottom()
    },
    
    handleRealtimeResult(data) {
      this.isTyping = false
      
      const botMessage = {
        id: this.messageId++,
        sender: 'bot',
        text: `Sentiment: ${data.sentiment} (Confidence: ${(data.confidence * 100).toFixed(1)}%)`,
        sentiment: {
          sentiment: data.sentiment,
          confidence: data.confidence,
          scores: data.scores
        },
        timestamp: data.timestamp
      }
      
      this.messages.push(botMessage)
      this.saveMessages()
      this.scrollToBottom()
    },
    
    handleError(data) {
      this.isTyping = false
      this.isSending = false
      
      const errorMessage = {
        id: this.messageId++,
        sender: 'bot',
        text: `Error: ${data.message}`,
        timestamp: new Date().toISOString()
      }
      
      this.messages.push(errorMessage)
      this.scrollToBottom()
    },
    
    handleKeyDown(event) {
      if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault()
        this.sendMessage()
      }
    },
    
    handleInput() {
      // Enviar indicador de escritura
      if (this.inputText.trim() && this.isConnected) {
        this.socket.emit('typing', { is_typing: true })
      }
    },
    
    formatTime(timestamp) {
      return new Date(timestamp).toLocaleTimeString()
    },
    
    scrollToBottom() {
      nextTick(() => {
        const container = this.$refs.messagesContainer
        if (container) {
          container.scrollTop = container.scrollHeight
        }
      })
    },
    
    saveMessages() {
      localStorage.setItem('sentiment_messages', JSON.stringify(this.messages))
    },
    
    loadStoredMessages() {
      const stored = localStorage.getItem('sentiment_messages')
      if (stored) {
        try {
          this.messages = JSON.parse(stored)
          this.messageId = Math.max(...this.messages.map(m => m.id), 0) + 1
          this.scrollToBottom()
        } catch (e) {
          console.error('Error loading stored messages:', e)
        }
      }
    }
  }
}
</script>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background-color: #f5f5f5;
  color: #333;
}

#app {
  display: flex;
  flex-direction: column;
  height: 100vh;
}

.header {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  padding: 1rem 0;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}

.header .container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 0 1rem;
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.header h1 {
  font-size: 1.5rem;
  font-weight: 600;
}

.connection-status {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  font-size: 0.9rem;
}

.status-indicator {
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background-color: #dc3545;
}

.status-indicator.connected {
  background-color: #28a745;
}

.main {
  flex: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

.container {
  max-width: 1200px;
  margin: 0 auto;
  padding: 1rem;
  height: 100%;
  display: flex;
  flex-direction: column;
}

.chat-container {
  flex: 1;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

.messages {
  flex: 1;
  padding: 1rem;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 1rem;
}

.message {
  display: flex;
  flex-direction: column;
  max-width: 70%;
}

.message.user {
  align-self: flex-end;
}

.message.bot {
  align-self: flex-start;
}

.message-content {
  background: #f8f9fa;
  padding: 1rem;
  border-radius: 12px;
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}

.message.user .message-content {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
}

.message-content p {
  margin-bottom: 0.5rem;
  line-height: 1.4;
}

.sentiment-info {
  display: flex;
  align-items: center;
  gap: 0.5rem;
  margin-top: 0.5rem;
}

.sentiment-badge {
  padding: 0.25rem 0.5rem;
  border-radius: 12px;
  font-size: 0.75rem;
  font-weight: 600;
  text-transform: uppercase;
}

.sentiment-badge.positivo {
  background-color: #d4edda;
  color: #155724;
}

.sentiment-badge.negativo {
  background-color: #f8d7da;
  color: #721c24;
}

.sentiment-badge.neutral {
  background-color: #fff3cd;
  color: #856404;
}

.confidence {
  font-size: 0.75rem;
  color: #666;
}

.message-time {
  font-size: 0.75rem;
  color: #666;
  margin-top: 0.25rem;
}

.message.user .message-time {
  align-self: flex-end;
}

.typing-indicator {
  display: flex;
  gap: 0.25rem;
  padding: 0.5rem;
}

.typing-indicator span {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background-color: #666;
  animation: typing 1.4s infinite ease-in-out;
}

.typing-indicator span:nth-child(2) {
  animation-delay: 0.2s;
}

.typing-indicator span:nth-child(3) {
  animation-delay: 0.4s;
}

@keyframes typing {
  0%, 80%, 100% {
    transform: scale(0.8);
    opacity: 0.5;
  }
  40% {
    transform: scale(1);
    opacity: 1;
  }
}

.input-container {
  padding: 1rem;
  background: white;
  border-radius: 8px;
  box-shadow: 0 2px 10px rgba(0,0,0,0.1);
  margin-top: 1rem;
}

.input-group {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
}

textarea {
  width: 100%;
  padding: 0.75rem;
  border: 2px solid #e9ecef;
  border-radius: 8px;
  font-family: inherit;
  font-size: 1rem;
  resize: vertical;
  min-height: 60px;
}

textarea:focus {
  outline: none;
  border-color: #667eea;
}

textarea:disabled {
  background-color: #f8f9fa;
  cursor: not-allowed;
}

.input-info {
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.char-count {
  font-size: 0.875rem;
  color: #666;
}

.send-button {
  background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
  color: white;
  border: none;
  padding: 0.75rem 1.5rem;
  border-radius: 8px;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
  transition: transform 0.2s;
}

.send-button:hover:not(:disabled) {
  transform: translateY(-1px);
}

.send-button:disabled {
  background: #6c757d;
  cursor: not-allowed;
  transform: none;
}

.footer {
  background: #343a40;
  color: white;
  padding: 1rem 0;
}

.stats {
  display: flex;
  justify-content: center;
  gap: 2rem;
}

.stat {
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 0.25rem;
}

.stat .label {
  font-size: 0.875rem;
  opacity: 0.8;
}

.stat .value {
  font-size: 1.25rem;
  font-weight: 600;
}

.stat .value.positive {
  color: #28a745;
}

.stat .value.negative {
  color: #dc3545;
}

.stat .value.neutral {
  color: #ffc107;
}

@media (max-width: 768px) {
  .header .container {
    flex-direction: column;
    gap: 1rem;
  }
  
  .message {
    max-width: 85%;
  }
  
  .stats {
    flex-wrap: wrap;
    gap: 1rem;
  }
  
  .stat {
    flex: 1;
    min-width: 80px;
  }
}
</style>
