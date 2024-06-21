import config

import tiktoken
import anthropic


ANTHROPIC_COMPLETION_OPTIONS = {
    "temperature": 0.7,
    "max_tokens": 4000,
    "top_p": 1,
    "timeout": 60.0
}


class Claude:
    def __init__(self, model="claude-3-5-sonnet-20240620"):
        assert model in {"claude-3-opus-20240229","claude-3-sonnet-20240229","claude-3-5-sonnet-20240620"}, f"Unknown model: {model}"
        self.model = model
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)

    async def send_message(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"claude-3-opus-20240229","claude-3-sonnet-20240229","claude-3-5-sonnet-20240620"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r = await self.client.messages.create(
                        model=self.model,
                        system = config.chat_modes[chat_mode]["prompt_start"],
                        messages=messages,
                        **ANTHROPIC_COMPLETION_OPTIONS
                    )
                    answer = r.content[0].text
                else:
                    raise ValueError(f"Unknown model: {self.model}")

                answer = self._postprocess_answer(answer)
                n_input_tokens, n_output_tokens = r.usage.input_tokens, r.usage.output_tokens
            except anthropic.APIStatusError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e
                if e.status_code =='403':                     
                    # forget first message in dialog_messages
                    dialog_messages = dialog_messages[1:]
                elif e.status_code == '429': # 達到限速上限
                    raise ValueError(f"達到使用限制上限，請稍後再試試看: {e}")


        n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)

        return answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed

    async def send_message_stream(self, message, dialog_messages=[], chat_mode="assistant"):
        if chat_mode not in config.chat_modes.keys():
            raise ValueError(f"Chat mode {chat_mode} is not supported")

        n_dialog_messages_before = len(dialog_messages)
        answer = None
        while answer is None:
            try:
                if self.model in {"claude-3-opus-20240229","claude-3-sonnet-20240229","claude-3-5-sonnet-20240620"}:
                    messages = self._generate_prompt_messages(message, dialog_messages, chat_mode)
                    r_gen = await self.client.messages.create(
                        model=self.model,
                        system = config.chat_modes[chat_mode]["prompt_start"],
                        messages=messages,
                        stream=True,
                        **ANTHROPIC_COMPLETION_OPTIONS
                    )

                    answer = ""
                    async for r_item in r_gen:
                        if r_item.type == "content_block_delta":
                            answer += r_item.delta.text
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed
                        elif r_item.type == 'message_delta':
                            n_input_tokens, n_output_tokens = self._count_tokens_from_messages(messages, answer, model=self.model)
                            n_output_tokens = r_item.usage.output_tokens
                            n_first_dialog_messages_removed = n_dialog_messages_before - len(dialog_messages)
                            yield "not_finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed 

                answer = self._postprocess_answer(answer)

            except anthropic.APIStatusError as e:  # too many tokens
                if len(dialog_messages) == 0:
                    raise ValueError("Dialog messages is reduced to zero, but still has too many tokens to make completion") from e
                if e.status_code =='403':                     
                    # forget first message in dialog_messages
                    dialog_messages = dialog_messages[1:]
                elif e.status_code == '429': # 達到限速上限
                    raise ValueError(f"達到使用限制上限，請稍後再試試看: {e}")
            
        yield "finished", answer, (n_input_tokens, n_output_tokens), n_first_dialog_messages_removed  # sending final answer

    def _generate_prompt(self, message, dialog_messages, chat_mode):
        prompt = config.chat_modes[chat_mode]["prompt_start"]
        prompt += "\n\n"

        # add chat context
        if len(dialog_messages) > 0:
            prompt += "Chat:\n"
            for dialog_message in dialog_messages:
                prompt += f"User: {dialog_message['user']}\n"
                prompt += f"Assistant: {dialog_message['bot']}\n"

        # current message
        prompt += f"User: {message}\n"
        prompt += "Assistant: "

        return prompt

    def _generate_prompt_messages(self, message, dialog_messages, chat_mode):

        messages = []
        for dialog_message in dialog_messages:
            messages.append({"role": "user", "content": dialog_message["user"]})
            messages.append({"role": "assistant", "content": dialog_message["bot"]})
        messages.append({"role": "user", "content": message})

        return messages

    def _postprocess_answer(self, answer):
        answer = answer.strip()
        return answer

    def _count_tokens_from_messages(self, messages, answer, model="claude-3-5-sonnet-20240620"):
        if 'claude' in  model:
            encoding = tiktoken.encoding_for_model('gpt-4')
        else:
            encoding = tiktoken.encoding_for_model(model)
        if model == "gpt-3.5-turbo-16k":
            tokens_per_message = 4  # every message follows <im_start>{role/name}\n{content}<im_end>\n
            tokens_per_name = -1  # if there's a name, the role is omitted
        elif model == "gpt-3.5-turbo":
            tokens_per_message = 4
            tokens_per_name = -1
        elif model == "gpt-4":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "gpt-4-1106-preview":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "claude-3-opus-20240229":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "claude-3-sonnet-20240229":
            tokens_per_message = 3
            tokens_per_name = 1
        elif model == "claude-3-5-sonnet-20240620":
            tokens_per_message = 3
            tokens_per_name = 1
        else:
            raise ValueError(f"Unknown model: {model}")

        # input
        n_input_tokens = 0
        for message in messages:
            n_input_tokens += tokens_per_message
            for key, value in message.items():
                n_input_tokens += len(encoding.encode(value))
                if key == "name":
                    n_input_tokens += tokens_per_name

        n_input_tokens += 2

        # output
        n_output_tokens = 1 + len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens

    def _count_tokens_from_prompt(self, prompt, answer, model="text-davinci-003"):
        encoding = tiktoken.encoding_for_model(model)

        n_input_tokens = len(encoding.encode(prompt)) + 1
        n_output_tokens = len(encoding.encode(answer))

        return n_input_tokens, n_output_tokens
