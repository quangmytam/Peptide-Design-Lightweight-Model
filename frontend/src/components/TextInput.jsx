import React from 'react';

const TextInput = ({ label, type = 'text', defaultValue, placeholder }) => {
  return (
    <label className="flex flex-col">
      <p className="text-sm font-medium pb-2">{label}</p>
      <input
        className="form-input w-full rounded-lg text-text-light dark:text-text-dark focus:outline-none focus:ring-2 focus:ring-primary/50 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark h-12 p-3 text-base"
        type={type}
        defaultValue={defaultValue}
        placeholder={placeholder}
      />
    </label>
  );
};

export default TextInput;
