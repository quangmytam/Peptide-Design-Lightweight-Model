import React, { useContext } from 'react';
import { NavLink, Link } from 'react-router-dom';
import { useTranslation } from 'react-i18next';
import { ThemeContext } from '../contexts/ThemeContext';
import Button from './Button';

const Header = () => {
  const { theme, toggleTheme } = useContext(ThemeContext);
  const { t, i18n } = useTranslation();

  const handleLanguageToggle = () => {
    const newLang = i18n.language.startsWith('en') ? 'vi' : 'en';
    i18n.changeLanguage(newLang);
  };

  const navLinks = [
    { to: '/datasets', label: t('nav.datasets') },
    { to: '/training', label: t('nav.training') },
    { to: '/evaluation', label: t('nav.evaluation') },
    { to: '/generation', label: t('nav.generation') },
    { to: '/documentation', label: t('nav.documentation') },
  ];

  return (
    <header className="flex items-center justify-between whitespace-nowrap border-b border-slate-900/10 dark:border-slate-50/10 px-4 py-4 md:px-6">
      <Link to="/" className="flex items-center gap-3">

        <h2 className="text-text-light dark:text-text-dark text-xl font-black tracking-widest">{t('header.logo')}</h2>
      </Link>
      <nav className="hidden items-center gap-8 md:flex">
        {navLinks.map(link => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              isActive
                ? 'text-sm font-medium text-primary hover:opacity-80'
                : 'text-sm font-medium text-slate-600 dark:text-slate-300 hover:text-primary dark:hover:text-primary'
            }
          >
            {link.label}
          </NavLink>
        ))}
      </nav>
      <div className="flex items-center gap-2">
        <Button onClick={handleLanguageToggle} variant="outline" className="w-10 h-10 font-bold dark:border-slate-700">
          {i18n.language.startsWith('vi') ? 'VI' : 'EN'}
        </Button>
        <Button onClick={toggleTheme} variant="icon" className="w-10 h-10">
          <span className="material-symbols-outlined">
            {theme === 'dark' ? 'light_mode' : 'dark_mode'}
          </span>
        </Button>
      </div>
    </header>
  );
};

export default Header;
