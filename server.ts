import 'dotenv/config';
import {
  GenerateContentResponse,
  GoogleGenAI,
  LiveServerMessage,
  Modality,
  ThinkingLevel,
} from '@google/genai';
import express, { Request, Response } from 'express';
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';
import { WebSocket, WebSocketServer } from 'ws';
import type { LanguageCode } from './src/types/schemes.ts';
import {
  buildAdvisorSystemInstruction,
  resolveTaskModel,
  sanitizeGroundingSources,
  ThinkingModeName,
  validateChatPayload,
} from './src/utils/aiAdvisorHelpers.ts';

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const PORT = Number(process.env.PORT) || 3000;

function getGeminiClient(): GoogleGenAI {
  const apiKey = process.env.GEMINI_API_KEY?.trim();
  if (!apiKey || apiKey === 'MY_GEMINI_API_KEY') {
    throw Object.assign(
      new Error('GEMINI_API_KEY is not configured. Please verify Settings > Secrets.'),
      { statusCode: 503 },
    );
  }
  return new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        'User-Agent': 'aistudio-build',
      },
    },
  });
}

function toThinkingLevel(name: ThinkingModeName): ThinkingLevel {
  switch (name) {
    case 'MINIMAL':
      return ThinkingLevel.MINIMAL;
    case 'LOW':
      return ThinkingLevel.LOW;
    case 'HIGH':
      return ThinkingLevel.HIGH;
    case 'MEDIUM':
    default:
      return ThinkingLevel.MEDIUM;
  }
}

function extractReadableErrorMessage(rawMsg: string): string {
  const trimmed = rawMsg.trim();
  if (trimmed.startsWith('{')) {
    try {
      const parsed = JSON.parse(trimmed) as { error?: { message?: string }; message?: string };
      if (typeof parsed?.error?.message === 'string' && parsed.error.message.trim()) {
        return parsed.error.message.trim();
      }
      if (typeof parsed?.message === 'string' && parsed.message.trim()) {
        return parsed.message.trim();
      }
    } catch {
      // Keep raw string if not JSON
    }
  }
  return trimmed;
}

function formatGeminiError(err: unknown): { status: number; message: string } {
  if (err && typeof err === 'object') {
    const anyErr = err as { status?: number; statusCode?: number; message?: string };
    const status =
      typeof anyErr.statusCode === 'number'
        ? anyErr.statusCode
        : typeof anyErr.status === 'number'
          ? anyErr.status
          : 500;
    const rawMsg = typeof anyErr.message === 'string' ? anyErr.message : 'Gemini request failed.';
    const cleanMsg = extractReadableErrorMessage(rawMsg);

    if (
      status === 401 ||
      status === 403 ||
      rawMsg.includes('API_KEY_INVALID') ||
      rawMsg.includes('PERMISSION_DENIED')
    ) {
      return {
        status: 403,
        message: 'Gemini API authorization failed. Please check GEMINI_API_KEY in Settings > Secrets.',
      };
    }
    if (status === 429 || rawMsg.includes('RESOURCE_EXHAUSTED')) {
      return {
        status: 429,
        message: 'Gemini quota limit reached. Please wait a moment and try again.',
      };
    }
    if (status === 503 || rawMsg.includes('"code":503') || rawMsg.includes('UNAVAILABLE')) {
      return {
        status: 503,
        message: cleanMsg || 'The Gemini model is temporarily busy. Please try again in a few seconds.',
      };
    }
    return { status: status >= 400 && status < 600 ? status : 500, message: cleanMsg };
  }
  return { status: 500, message: 'Unexpected server error while contacting Gemini.' };
}

async function withTransientRetry<T>(fn: () => Promise<T>, maxRetries = 2): Promise<T> {
  let lastError: unknown;
  for (let attempt = 0; attempt <= maxRetries; attempt++) {
    try {
      return await fn();
    } catch (err) {
      lastError = err;
      const anyErr = err as { status?: number; message?: string };
      const status = anyErr?.status;
      const msg = typeof anyErr?.message === 'string' ? anyErr.message : '';
      const isTransient =
        status === 503 || status === 504 || msg.includes('"code":503') || msg.includes('UNAVAILABLE');
      if (attempt < maxRetries && isTransient) {
        await new Promise((r) => setTimeout(r, 600 * (attempt + 1)));
        continue;
      }
      throw err;
    }
  }
  throw lastError;
}

async function startServer() {
  const app = express();
  app.use(express.json({ limit: '1mb' }));

  app.get('/api/health', (_req: Request, res: Response) => {
    const hasKey =
      Boolean(process.env.GEMINI_API_KEY?.trim()) &&
      process.env.GEMINI_API_KEY?.trim() !== 'MY_GEMINI_API_KEY';
    res.json({ ok: true, geminiConfigured: hasKey });
  });

  app.post('/api/chat', async (req: Request, res: Response) => {
    const validation = validateChatPayload(req.body);
    if (!validation.ok) {
      res.status(400).json({ error: validation.error });
      return;
    }

    const { messages, mode, language, useSearchGrounding } = validation.data;
    const route = resolveTaskModel(mode);
    const model = useSearchGrounding ? 'gemini-4-hydrogen' : route.model;

    try {
      const ai = getGeminiClient();
      const systemInstruction = buildAdvisorSystemInstruction(mode, language);
      const priorHistory = messages.slice(0, -1).map((turn) => ({
        role: turn.role,
        parts: [{ text: turn.text }],
      }));
      const latestTurn = messages[messages.length - 1];

      const config: {
        systemInstruction: string;
        thinkingConfig: { thinkingLevel: ThinkingLevel };
        tools?: Array<{ googleSearch: Record<string, never> }>;
      } = {
        systemInstruction,
        thinkingConfig: {
          thinkingLevel: toThinkingLevel(useSearchGrounding ? 'LOW' : route.thinkingLevel),
        },
      };

      if (useSearchGrounding) {
        config.tools = [{ googleSearch: {} }];
      }

      const response: GenerateContentResponse = await withTransientRetry(async () => {
        const chat = ai.chats.create({
          model,
          history: priorHistory,
          config,
        });
        return chat.sendMessage({ message: latestTurn.text });
      });

      const sources = sanitizeGroundingSources(
        response.candidates?.[0]?.groundingMetadata?.groundingChunks,
      );

      res.json({
        reply: response.text ?? '',
        modelUsed: model,
        thinkingLevel: useSearchGrounding ? 'LOW' : route.thinkingLevel,
        sources,
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  app.post('/api/search-grounding', async (req: Request, res: Response) => {
    const rawQuery = typeof req.body?.query === 'string' ? req.body.query.trim() : '';
    const language: LanguageCode =
      req.body?.language === 'hi' || req.body?.language === 'kn' ? req.body.language : 'en';

    if (rawQuery.length < 3) {
      res.status(400).json({ error: 'Please enter a scheme question of at least 3 characters.' });
      return;
    }

    try {
      const ai = getGeminiClient();
      const response: GenerateContentResponse = await withTransientRetry(() =>
        ai.models.generateContent({
          model: 'gemini-4-hydrogen',
          contents: `Check the latest official Indian government scheme rules, updates, premiums, or enrollment guidelines for: ${rawQuery.slice(0, 400)}`,
          config: {
            systemInstruction: buildAdvisorSystemInstruction('search', language),
            tools: [{ googleSearch: {} }],
          },
        }),
      );

      const sources = sanitizeGroundingSources(
        response.candidates?.[0]?.groundingMetadata?.groundingChunks,
      );

      res.json({
        answer: response.text ?? '',
        sources,
        modelUsed: 'gemini-4-hydrogen',
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      res.status(formatted.status).json({ error: formatted.message });
    }
  });

  const isProd = process.env.NODE_ENV === 'production';
  if (!isProd) {
    const { createServer: createViteServer } = await import('vite');
    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: 'spa',
    });
    app.use(vite.middlewares);
  } else {
    const distDir = path.resolve(__dirname, 'dist');
    app.use(express.static(distDir));
    app.get('*', (_req: Request, res: Response) => {
      const indexFile = path.join(distDir, 'index.html');
      if (fs.existsSync(indexFile)) {
        res.sendFile(indexFile);
      } else {
        res.status(404).send('Build output not found.');
      }
    });
  }

  const httpServer = http.createServer(app);
  const wss = new WebSocketServer({ noServer: true });

  httpServer.on('upgrade', (request, socket, head) => {
    const urlPath = request.url?.split('?')[0];
    if (urlPath === '/live') {
      wss.handleUpgrade(request, socket, head, (ws) => {
        wss.emit('connection', ws, request);
      });
    } else {
      socket.destroy();
    }
  });

  wss.on('connection', async (clientWs: WebSocket, request: http.IncomingMessage) => {
    const parsedUrl = new URL(request.url ?? '/live', 'http://localhost');
    const langParam = parsedUrl.searchParams.get('lang');
    const language: LanguageCode =
      langParam === 'hi' || langParam === 'kn' ? langParam : 'en';

    let sessionClosed = false;
    let session: Awaited<ReturnType<GoogleGenAI['live']['connect']>> | null = null;

    const safeSend = (payload: Record<string, unknown>) => {
      if (clientWs.readyState === WebSocket.OPEN) {
        clientWs.send(JSON.stringify(payload));
      }
    };

    try {
      const ai = getGeminiClient();
      session = await ai.live.connect({
        model: 'gemini-3.8-live',
        config: {
          responseModalities: [Modality.AUDIO],
          speechConfig: {
            voiceConfig: {
              prebuiltVoiceConfig: { voiceName: 'Zephyr' },
            },
          },
          inputAudioTranscription: {},
          outputAudioTranscription: {},
          systemInstruction: buildAdvisorSystemInstruction('voice', language),
        },
        callbacks: {
          onopen: () => {
            safeSend({ type: 'ready', model: 'gemini-3.8-live' });
          },
          onmessage: (message: LiveServerMessage) => {
            const content = message.serverContent;
            if (!content) return;

            if (content.inputTranscription?.text) {
              safeSend({
                type: 'inputTranscript',
                text: content.inputTranscription.text,
              });
            }

            if (content.outputTranscription?.text) {
              safeSend({
                type: 'outputTranscript',
                text: content.outputTranscription.text,
              });
            }

            for (const part of content.modelTurn?.parts ?? []) {
              if (part.inlineData?.data) {
                safeSend({
                  type: 'audio',
                  data: part.inlineData.data,
                });
              }
            }

            if (content.interrupted) {
              safeSend({ type: 'interrupted' });
            }

            if (content.turnComplete) {
              safeSend({ type: 'turnComplete' });
            }
          },
          onerror: (event: ErrorEvent) => {
            safeSend({
              type: 'error',
              message: event?.message || 'Live voice session encountered a connection error.',
            });
          },
          onclose: () => {
            if (!sessionClosed) {
              safeSend({ type: 'closed' });
            }
          },
        },
      });
    } catch (err) {
      const formatted = formatGeminiError(err);
      safeSend({ type: 'error', message: formatted.message });
      clientWs.close();
      return;
    }

    clientWs.on('message', (raw) => {
      if (!session || sessionClosed) return;
      try {
        const parsed = JSON.parse(raw.toString()) as { audio?: string; text?: string };
        if (typeof parsed.audio === 'string' && parsed.audio.length > 0) {
          session.sendRealtimeInput({
            audio: { data: parsed.audio, mimeType: 'audio/pcm;rate=16000' },
          });
        } else if (typeof parsed.text === 'string' && parsed.text.trim().length > 0) {
          session.sendRealtimeInput({
            text: parsed.text.trim().slice(0, 1000),
          });
        }
      } catch {
        safeSend({ type: 'error', message: 'Invalid WebSocket message payload.' });
      }
    });

    clientWs.on('close', () => {
      sessionClosed = true;
      if (session) {
        try {
          session.close();
        } catch {
          // Ignore cleanup error
        }
      }
    });
  });

  httpServer.listen(PORT, '0.0.0.0', () => {
    console.log(`Suraksha Setu server listening on http://localhost:${PORT}`);
  });
}

startServer().catch((err) => {
  console.error('Failed to start server:', err);
  process.exit(1);
});
