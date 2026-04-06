import os
import requests
import json


class SkincareChatbot:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get('OPENAI_API_KEY', '')
        self.api_url = 'https://api.openai.com/v1/chat/completions'
        self.system_prompt = """You are Lumina, a friendly and knowledgeable AI skincare assistant.
You help users understand their skin conditions and provide practical advice.

Your role:
- Explain skin conditions in simple, reassuring language
- Recommend general skincare routines and ingredients that may help
- Suggest dietary and lifestyle changes that support skin health
- Always recommend consulting a dermatologist for persistent or severe conditions
- Be empathetic and supportive

You are NOT a doctor. Always include a disclaimer that your advice is general
and not a substitute for professional medical advice."""

    def get_diagnosis_context(self, diagnosis):
        if not diagnosis:
            return ''

        condition = diagnosis.get('prediction', '')
        confidence = diagnosis.get('confidence', 0)

        return f"""The user has uploaded a skin image. The AI analysis detected:
- Condition: {condition}
- Confidence: {confidence}%

Provide advice based on this analysis. If confidence is below 70%, mention that
the result is uncertain and recommend a professional evaluation."""

    def chat(self, user_message, diagnosis=None, conversation_history=None):
        if not self.api_key:
            return self._fallback_response(user_message, diagnosis)

        messages = [{'role': 'system', 'content': self.system_prompt}]

        if diagnosis:
            context = self.get_diagnosis_context(diagnosis)
            messages.append({'role': 'system', 'content': context})

        if conversation_history:
            messages.extend(conversation_history)

        messages.append({'role': 'user', 'content': user_message})

        try:
            response = requests.post(
                self.api_url,
                headers={
                    'Authorization': f'Bearer {self.api_key}',
                    'Content-Type': 'application/json'
                },
                json={
                    'model': 'gpt-4o-mini',
                    'messages': messages,
                    'max_tokens': 500,
                    'temperature': 0.7
                },
                timeout=30
            )

            if response.status_code == 200:
                data = response.json()
                reply = data['choices'][0]['message']['content']
                return {'reply': reply, 'status': 'success'}
            else:
                return self._fallback_response(user_message, diagnosis)

        except Exception as e:
            return self._fallback_response(user_message, diagnosis)

    def _fallback_response(self, user_message, diagnosis=None):
        condition = diagnosis.get('prediction', 'your skin condition') if diagnosis else 'your skin condition'
        confidence = diagnosis.get('confidence', 0) if diagnosis else 0

        responses = {
            'acne': {
                'description': 'Acne is a common skin condition caused by clogged pores, bacteria, and excess oil production.',
                'tips': [
                    'Use a gentle cleanser twice daily',
                    'Look for products with salicylic acid or benzoyl peroxide',
                    'Avoid touching your face frequently',
                    'Use non-comedogenic (non-pore-clogging) moisturisers and sunscreen',
                    'Stay hydrated and maintain a balanced diet rich in fruits and vegetables',
                    'Consider reducing dairy and high-sugar foods which may trigger breakouts'
                ]
            },
            'Vitiligo': {
                'description': 'Vitiligo is a condition where patches of skin lose their pigment. It occurs when melanocytes (pigment-producing cells) are destroyed.',
                'tips': [
                    'Protect affected areas from sun exposure with SPF 30+ sunscreen',
                    'Consider cosmetic concealers designed for vitiligo if desired',
                    'Eat foods rich in antioxidants (berries, leafy greens, nuts)',
                    'Vitamin D and B12 supplements may support skin health (consult your doctor)',
                    'Stress management can help as stress may trigger flare-ups',
                    'Phototherapy is a common medical treatment - discuss with a dermatologist'
                ]
            },
            'hyperpigmentation': {
                'description': 'Hyperpigmentation occurs when patches of skin become darker than surrounding areas due to excess melanin production.',
                'tips': [
                    'Daily sunscreen (SPF 30+) is essential to prevent darkening',
                    'Look for products with vitamin C, niacinamide, or alpha arbutin',
                    'Chemical exfoliants (AHAs like glycolic acid) can help fade dark spots',
                    'Retinoids can improve skin cell turnover (start slowly)',
                    'Avoid picking or scratching affected areas',
                    'Eat foods rich in vitamin C and E to support skin repair'
                ]
            }
        }

        info = responses.get(condition, {
            'description': 'I detected a skin condition but need more context to provide specific advice.',
            'tips': ['Please consult a dermatologist for an accurate diagnosis.']
        })

        tips_text = '\n'.join(f'- {tip}' for tip in info['tips'])

        reply = f"""{info['description']}

Here are some recommendations:
{tips_text}

Please note: This advice is general and not a substitute for professional medical advice. If your condition persists or worsens, please consult a dermatologist."""

        return {'reply': reply, 'status': 'fallback'}
