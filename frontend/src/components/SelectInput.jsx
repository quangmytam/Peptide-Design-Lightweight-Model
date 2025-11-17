import React from 'react';

const SelectInput = ({ label, options }) => {
  return (
    <label className="flex flex-col">
      <p className="text-base font-medium pb-2">{label}</p>
      <div className="relative">
        <select className="form-input w-full resize-none overflow-hidden rounded-lg text-text-light dark:text-text-dark focus:outline-none focus:ring-2 focus:ring-primary/50 border border-border-light dark:border-border-dark bg-surface-light dark:bg-surface-dark h-12 p-3 text-base font-normal bg-no-repeat bg-right-3" style={{ backgroundImage: "url('data:image/svg+xml,%3csvg xmlns=%27http://www.w3.org/2000/svg%27 width=%2724px%27 height=%2724px%27 fill=%27%239ca3af%27 viewBox=%270 0 256 256%27%3e%3cpath d=%27M215.39 92.94a8 8 0 00-11.31 0L128 169.37 51.92 93.3a8 8 0 00-11.32 11.31l82.05 82.06a8 8 0 0011.32 0l81.42-81.42a8 8 0 000-11.31z%27%3e%3c/path%3e%3c/svg%3e')" }}>
          {options.map((option) => (
            <option key={option}>{option}</option>
          ))}
        </select>
      </div>
    </label>
  );
};

export default SelectInput;
