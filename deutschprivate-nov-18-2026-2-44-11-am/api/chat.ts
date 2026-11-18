import { GoogleGenAI, ThinkingLevel } from '@google/genai';

export default async function handler(req: any, res: any) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  if (req.method !== 'POST') {
    return res.status(405).json({ error: 'Method Not Allowed (405)', code: 405 });
  }

  const apiKey = String(process.env.VITE_GEMINI_API_KEY || process.env.GEMINI_API_KEY || '')
    .trim()
    .replace(/^["']|["']$/g, '')
    .trim();

  if (!apiKey) {
    return res.status(500).json({
      error: 'Missing API Key (GEMINI_API_KEY chưa được cấu hình trên Vercel/Server).',
      code: 'Missing API Key',
    });
  }

  try {
    const body = typeof req.body === 'string' ? JSON.parse(req.body || '{}') : req.body || {};
    const { messages, roleKey, customInstructions } = body;

    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ error: 'Vui lòng nhập tin nhắn để bắt đầu trò chuyện.', code: 400 });
    }

    const roleInstructions: Record<string, string> = {
      tutor:
        'Bạn là một người bạn học / đồng nghiệp bản xứ Đức thân thiện, hướng dẫn tiếng Đức trực tiếp, tự nhiên như đang trò chuyện hàng ngày với người Việt (A2–C1). ' +
        'QUY TẮC ĐỊNH DẠNG BẮT BUỘC:\n' +
        '- Tuyệt đối KHÔNG dùng tiêu đề lớn (#, ##, ###), KHÔNG chia thành các mục kiểu giáo án (như 1. Thông tin từ vựng, 2. Cách dùng thực tế, 3. Lưu ý) và KHÔNG giải thích dài dòng rập khuôn.\n' +
        '- Khi người dùng hỏi một từ hoặc cụm từ tiếng Đức (ví dụ: "profie"), hãy trả lời ngắn gọn thành 2–3 dòng súc tích: sửa chính tả (nếu sai) + ghi thẳng dạng chuẩn kèm mạo từ/số nhiều (ví dụ: der Profi, die Profis), giải thích nghĩa tiếng Việt tự nhiên trong 1 câu, và đưa 1–2 câu ví dụ giao tiếp thực tế (kèm dịch nghĩa) ngay bên dưới.\n' +
        '- Với câu hỏi ngữ pháp hoặc diễn đạt, trả lời trực tiếp vào trọng tâm, ngắn gọn, tự nhiên kèm ví dụ thực tế.',
      partner:
        'Du bist ein freundlicher deutscher Gesprächspartner für Lernende auf Niveau B1–B2. Antworte überwiegend auf natürlichem Alltagsdeutsch in kurzen, klaren Sätzen (ohne Markdown-Überschriften #, ##, ### oder nummerierte Listen) und stelle am Ende eine passende Rückfrage. Wenn der Lernende einen Fehler macht, füge am Ende in Klammern eine kurze, höfliche Korrektur auf Vietnamesisch hinzu.',
      exam:
        'Bạn là Huấn luyện viên luyện thi Goethe-Zertifikat & telc (B1–C1). Hỗ trợ học viên luyện Schreiben và Sprechen bằng các mẫu câu Redemittel chuẩn kỳ thi, sửa lỗi câu trực tiếp và ngắn gọn. Tuyệt đối KHÔNG dùng tiêu đề lớn (#, ##, ###) hay chia mục giáo án rườm rà.',
    };

    const baseInstruction = roleInstructions[String(roleKey || 'tutor')] || roleInstructions.tutor;
    const cleanCustomRules = Array.isArray(customInstructions)
      ? customInstructions
          .map((r: any) => String(r || '').trim())
          .filter((r: string) => r.length > 0)
          .slice(0, 30)
      : [];

    const fullSystemInstruction =
      cleanCustomRules.length > 0
        ? `${baseInstruction}\n\nChỉ dẫn bổ sung từ người dùng (Bắt buộc tuân thủ):\n${cleanCustomRules
            .map((r, i) => `${i + 1}. ${r}`)
            .join('\n')}`
        : baseInstruction;

    const sanitized = messages
      .map((m: any) => ({
        role: m?.role === 'model' ? 'model' : 'user',
        text: String(m?.text || '').trim(),
      }))
      .filter((m: any) => m.text.length > 0)
      .slice(-10);

    const normalizedWindow: Array<{ role: 'user' | 'model'; text: string }> = [];
    for (const item of sanitized) {
      const last = normalizedWindow[normalizedWindow.length - 1];
      if (last && last.role === item.role) {
        last.text = `${last.text}\n${item.text}`;
      } else {
        normalizedWindow.push({ role: item.role as 'user' | 'model', text: item.text });
      }
    }

    const latestMsg = normalizedWindow[normalizedWindow.length - 1];
    if (!latestMsg || latestMsg.role !== 'user') {
      return res.status(400).json({ error: 'Tin nhắn cuối cùng phải từ người dùng.', code: 400 });
    }

    const contents = normalizedWindow.map((m) => ({
      role: m.role,
      parts: [{ text: m.text }],
    }));

    const ai = new GoogleGenAI({
      apiKey,
      httpOptions: {
        headers: {
          'User-Agent': 'aistudio-build',
        },
      },
    });

    const primaryModels = ['gemini-3.1-flash-lite-preview', 'gemini-3.1-flash-lite', 'gemini-4-hydrogen'];
    const fallbackModels = ['gemini-2.5-flash-lite', 'gemini-2.5-flash'];
    const retryDelays = [1500, 3000, 6000];

    const is503Err = (err: any): boolean => {
      const st = Number(err?.status || err?.code || err?.error?.code || 0);
      if (st === 503) return true;
      const msg = String(err?.message || err?.error?.message || '').toLowerCase();
      const statusText = String(err?.error?.status || '').toUpperCase();
      return (
        statusText === 'UNAVAILABLE' ||
        msg.includes('503') ||
        msg.includes('unavailable') ||
        msg.includes('high demand') ||
        msg.includes('overloaded')
      );
    };

    const is404Err = (err: any): boolean => {
      const st = Number(err?.status || err?.code || 0);
      if (st === 404) return true;
      const msg = String(err?.message || '').toLowerCase();
      return msg.includes('not_found') || msg.includes('not found');
    };

    const callModel = async (modelName: string) => {
      const cfg: Record<string, any> = {
        systemInstruction: fullSystemInstruction,
      };
      if (!modelName.startsWith('gemini-2.5-') && modelName !== 'gemini-4-hydrogen') {
        cfg.thinkingConfig = { thinkingLevel: ThinkingLevel.MINIMAL };
      } else if (modelName === 'gemini-4-hydrogen') {
        cfg.thinkingConfig = { thinkingLevel: ThinkingLevel.LOW };
      }
      try {
        return await ai.models.generateContent({
          model: modelName,
          contents,
          config: cfg,
        });
      } catch (err: any) {
        const msg = String(err?.message || '');
        if (err?.status === 400 && (msg.includes('thinking') || msg.includes('system_instruction'))) {
          const prefix = `[System Instruction]\n${fullSystemInstruction}\n\n`;
          const fallbackContents = contents.map((c, idx) =>
            idx === 0 ? { ...c, parts: [{ text: prefix + c.parts[0].text }] } : c
          );
          return await ai.models.generateContent({
            model: modelName,
            contents: fallbackContents,
          });
        }
        throw err;
      }
    };

    let response: any = null;
    let lastError: any = null;
    let activeModel = primaryModels[0];
    let usedFallback = false;
    let retryCount = 0;

    for (const modelName of primaryModels) {
      activeModel = modelName;
      while (true) {
        try {
          response = await callModel(modelName);
          break;
        } catch (err: any) {
          lastError = err;
          if (is404Err(err)) break;
          if (is503Err(err)) {
            if (retryCount < retryDelays.length) {
              await new Promise((r) => setTimeout(r, retryDelays[retryCount]));
              retryCount++;
              continue;
            }
            break;
          }
          throw err;
        }
      }
      if (response) break;
      if (retryCount >= retryDelays.length && is503Err(lastError)) break;
    }

    if (!response) {
      for (const fbModel of fallbackModels) {
        activeModel = fbModel;
        try {
          response = await callModel(fbModel);
          usedFallback = true;
          break;
        } catch (err: any) {
          lastError = err;
          if (is503Err(err) || is404Err(err)) continue;
          throw err;
        }
      }
    }

    if (!response) {
      if (lastError && typeof lastError === 'object') lastError.retriedOnServer = true;
      throw lastError || new Error('Không thể kết nối mô hình Gemini.');
    }

    const reply = (response.text || '').trim();
    if (!reply) {
      return res.status(502).json({ error: 'Mô hình AI không trả về nội dung.', code: 502 });
    }

    return res.status(200).json({ reply, model: activeModel, usedFallback });
  } catch (error: any) {
    const status =
      Number(error?.status || error?.code) === 503 ||
      String(error?.message || '').toLowerCase().includes('high demand') ||
      String(error?.message || '').toLowerCase().includes('unavailable')
        ? 503
        : error?.status && error.status >= 400 && error.status < 600
        ? error.status
        : 500;
    let cleanMsg = String(error?.message || 'Không thể kết nối Trợ lý AI lúc này.');
    if (cleanMsg.startsWith('{')) {
      try {
        const parsedErr = JSON.parse(cleanMsg);
        cleanMsg = parsedErr?.error?.message || cleanMsg;
      } catch {}
    }
    return res.status(status).json({
      error:
        status === 503
          ? 'Mô hình AI hiện đang quá tải tạm thời. Vui lòng thử gửi lại sau vài giây.'
          : cleanMsg,
      code: status,
      retriedOnServer: Boolean(error?.retriedOnServer),
    });
  }
}
