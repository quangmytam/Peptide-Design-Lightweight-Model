import React from 'react';

const PageTitle = ({ children }) => {
  return (
    <div className="mb-8">
      <h1 className="text-4xl font-black leading-tight tracking-tighter">{children}</h1>
    </div>
  );
};

export default PageTitle;
