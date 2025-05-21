import { createTheme, alpha } from '@mui/material/styles';

// Цветовая палитра по типу "Soft Blue-Purple"
const softBlue = '#5e7aa6';
const accentPurple = '#6A5ACD'; // Используется как primary.main
const lightGray = '#f7f8fc';
const darkGray = '#2c3e50'; // Используется как text.primary
const successGreen = '#2ecc71';
const warningYellow = '#f1c40f';
const errorRed = '#e74c3c';

// Кастомные тени
const softShadow = '0 4px 16px rgba(0, 0, 0, 0.06)';
const strongerSoftShadow = '0 8px 20px rgba(0, 0, 0, 0.08)';
const buttonShadow = '0 2px 8px rgba(0, 0, 0, 0.06)';
const buttonHoverShadow = '0 4px 12px rgba(0, 0, 0, 0.08)';

const theme = createTheme({
  palette: {
    mode: 'light',
    primary: {
      main: accentPurple,
      light: '#8d84e3', // Используем для hover эффектов, если нужно светлее
      dark: '#483d8b',  // Используем для hover эффектов, если нужно темнее
    },
    secondary: {
      main: softBlue,
    },
    success: {
      main: successGreen,
    },
    warning: {
      main: warningYellow,
    },
    error: {
      main: errorRed,
    },
    background: {
      default: lightGray,
      paper: '#ffffff',
    },
    text: {
      primary: darkGray,
      secondary: '#7f8c8d', // Для менее важного текста
    },
    divider: alpha(darkGray, 0.12), // Используем для границ
  },
  typography: {
    fontFamily: '"Inter", "Roboto", "Helvetica", "Arial", sans-serif',
    h1: {
      fontSize: '2.5rem',
      fontWeight: 700,
      // color: darkGray, // Заменено на text.primary ниже через callback
    },
    h2: {
      fontSize: '2rem',
      fontWeight: 600,
      // color: darkGray,
    },
    h3: {
      fontSize: '1.5rem',
      fontWeight: 600,
      // color: darkGray,
    },
    h4: {
      fontSize: '1.25rem',
      fontWeight: 600,
      // color: darkGray,
    },
    h5: {
      fontSize: '1rem',
      fontWeight: 600,
      // color: darkGray,
    },
    body1: {
      fontSize: '0.95rem',
      lineHeight: 1.75,
      color: '#34495e', // Оставил, т.к. это немного отличается от darkGray и может быть намеренно
                         // Если нет, можно заменить на (theme) => theme.palette.text.primary
    },
    button: {
      textTransform: 'none',
      fontWeight: 500,
    },
  },
  shape: {
    borderRadius: 8, // Базовый радиус
  },
  // Переопределяем стандартные тени MUI
  shadows: [
    'none',
    softShadow, // shadows[1]
    strongerSoftShadow, // shadows[2] - можно использовать для hover на карточках
    buttonShadow, // shadows[3] - для кнопок
    buttonHoverShadow, // shadows[4] - для hover кнопок
    // ... можно добавить еще до 24
  ].concat(
    Array(25 - 5).fill('none') // Заполняем остальные 'none' или другими значениями
  ) as any, // Приведение типа, т.к. MUI ожидает массив из 25 строк
  components: {
    // Применяем цвет текста ко всем заголовкам
    MuiTypography: {
      styleOverrides: {
        root: ({ ownerState, theme }) => ({
          ...(ownerState.variant && ['h1', 'h2', 'h3', 'h4', 'h5', 'h6'].includes(ownerState.variant) && {
            color: theme.palette.text.primary,
          }),
        }),
      },
    },
    MuiAppBar: {
      defaultProps: {
        elevation: 0,
        color: 'inherit',
      },
      styleOverrides: {
        root: ({ theme }) => ({
          backgroundColor: theme.palette.background.paper,
          borderBottom: `1px solid ${theme.palette.divider}`,
        }),
      },
    },
    MuiButton: {
      styleOverrides: {
        root: {
          boxShadow: 'none', // По умолчанию без тени для outlined/text
          transition: 'background-color 0.2s ease-in-out, color 0.2s ease-in-out, box-shadow 0.2s ease-in-out, transform 0.1s ease-in-out',
          '&:hover': {
            boxShadow: 'none',
            transform: 'none', // Убираем дефолтный transform: scale(1.0X) если есть
          },
        },
      },
      variants: [
        {
          props: { variant: 'contained', color: 'primary' },
          style: ({ theme }) => ({
            boxShadow: theme.shadows[3], // Используем нашу кастомную тень для кнопок
            color: theme.palette.primary.contrastText, // Автоматический контрастный текст
            '&:hover': {
              backgroundColor: theme.palette.primary.dark, // Немного темнее при наведении
              boxShadow: theme.shadows[4], // Усиленная тень при наведении
            },
          }),
        },
        {
          props: { variant: 'outlined' },
          style: ({ theme }) => ({
            borderWidth: '1px',
            borderColor: theme.palette.divider, // Более мягкая граница по умолчанию
            '&:hover': {
              borderWidth: '1px',
              borderColor: theme.palette.primary.main, // Акцентная граница при наведении
              backgroundColor: alpha(theme.palette.primary.main, 0.04), // Легкий фон основного цвета
            },
          }),
        },
        {
            props: { variant: 'text', color: 'primary' },
            style: ({ theme }) => ({
              '&:hover': {
                backgroundColor: alpha(theme.palette.primary.main, 0.08),
              },
            }),
          },
      ],
    },
    MuiPaper: {
      defaultProps: {
        elevation: 0, // Убираем дефолтную MUI тень, чтобы использовать свою
      },
      styleOverrides: {
        root: ({ theme }) => ({
          boxShadow: theme.shadows[1], // Используем нашу мягкую тень
          borderRadius: 12, // Немного больше, чем базовый
          backgroundImage: 'none', // Убираем градиенты, если они есть по умолчанию где-то
        }),
      },
    },
    MuiCard: {
      styleOverrides: {
        root: ({ theme }) => ({
          borderRadius: 12, // Как у MuiPaper
          boxShadow: theme.shadows[1], // Начальная тень
          transition: 'transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out',
          '&:hover': {
            transform: 'translateY(-2px)',
            boxShadow: theme.shadows[2], // Более сильная тень при наведении
          },
        }),
      },
    },
    MuiChip: {
      styleOverrides: {
        root: {
          fontWeight: 500,
          borderRadius: 6, // Чуть меньше базового для компактности
        },
      },
    },
    MuiTextField: {
      defaultProps: {
        variant: 'outlined',
        size: 'small',
        fullWidth: true,
      },
      styleOverrides: {
        root: {
          // borderRadius: 6, // MuiOutlinedInput ниже позаботится о радиусе
        },
      },
    },
    MuiInputLabel: {
      styleOverrides: {
        root: ({ theme }) => ({
          color: theme.palette.text.secondary, // Используем вторичный цвет текста
          '&.Mui-focused': {
            color: theme.palette.primary.main, // Основной цвет при фокусе
          },
        }),
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: ({ theme }) => ({
            borderRadius: theme.shape.borderRadius, // Используем базовый радиус
          '&:hover .MuiOutlinedInput-notchedOutline': {
            borderColor: alpha(theme.palette.primary.main, 0.5), // Граница при наведении
          },
          '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
            borderColor: theme.palette.primary.main,
            borderWidth: 1, // Можно сделать 2px для более явного фокуса, но 1px чище
            // boxShadow: `0 0 0 2px ${alpha(theme.palette.primary.main, 0.2)}`, // Вариант с "glow" эффектом при фокусе
          },
        }),
        notchedOutline: ({ theme }) => ({
          borderColor: theme.palette.divider, // Граница по умолчанию
        }),
      },
    },
    MuiIconButton: {
      styleOverrides: {
        root: ({ theme }) => ({
          transition: 'background-color 0.2s ease-in-out',
          '&:hover': {
            backgroundColor: alpha(theme.palette.primary.main, 0.08), // Легкий фон основного цвета
          },
        }),
      },
    },
    MuiTooltip: {
        styleOverrides: {
            tooltip: ({ theme }) => ({
                backgroundColor: alpha(theme.palette.text.primary, 0.92),
                fontSize: '0.75rem',
                borderRadius: theme.shape.borderRadius / 2,
            }),
            arrow: ({ theme }) => ({
                color: alpha(theme.palette.text.primary, 0.92),
            }),
        }
    },
    MuiListItemButton: {
        styleOverrides: {
            root: ({ theme }) => ({
                borderRadius: theme.shape.borderRadius,
                '&:hover': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.05),
                },
                '&.Mui-selected': {
                    backgroundColor: alpha(theme.palette.primary.main, 0.08),
                    '&:hover': {
                        backgroundColor: alpha(theme.palette.primary.main, 0.12),
                    }
                }
            })
        }
    }
  },
});

export default theme;